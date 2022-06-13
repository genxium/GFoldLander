# -*- coding: utf-8 -*-
from panda3d.core import loadPrcFileData 

"""
[WARNING] 

GUI thread draws lots of CPU and memory resources when running on a machine without dedicated GPU, set DESIRED_GUI_FPS to a low value to favor the computational power required by "runPid"!
"""
DESIRED_GUI_FPS = 60 

loadPrcFileData('', 'window-title OfflineGFold')
loadPrcFileData('', 'win-size 1024 768')
loadPrcFileData('', 'clock-mode limited')
loadPrcFileData('', f'clock-frame-rate {DESIRED_GUI_FPS}')
loadPrcFileData('', f'threading-model Cull/Draw') # Reference https://docs.panda3d.org/1.10/python/programming/rendering-process/multithreaded-render-pipeline

import colorlog as logging
import os
import sys

logFormatter = logging.ColoredFormatter(
    "%(log_color)s %(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]\n%(message)s")

rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.DEBUG)
rootLogger.propagate = False  # disables console logging if not later explicitly added

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

CWD = os.path.dirname(os.path.abspath(__file__))

rootLogger.info('CWD=%s', CWD)
sys.path.append(CWD)

import argparse, gc

import direct.directbase.DirectStart

from direct.showbase.DirectObject import DirectObject
from direct.showbase.InputStateGlobal import inputState

from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletDebugNode

from panda3d.core import Point3, Vec3, Vec4, LMatrix3, DirectionalLight, AmbientLight, AntialiasAttrib, BitMask32, Quat, TextNode

import math, numpy

from models.plant import LunarLanderFuel
from models.terrain import Terrain
from models.wind import Wind

from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.spatial import KDTree

import direct.stdpy.threading as threading # Use panda's lightweight threading library, see https://docs.panda3d.org/1.10/python/programming/tasks-and-events/task-chains
from panda3d.core import TP_low, TP_normal, TP_high, TP_urgent

import gevent
from gevent.event import Event
from gevent.lock import BoundedSemaphore
gfoldMux = BoundedSemaphore(1)
 
import cvxpy as opt

import pickle
 
import time

from pympler import muppy, summary, tracker
objtracker = tracker.ObjectTracker()
smytracker = tracker.SummaryTracker()
prevObjs = None
currObjs = None

args = None

epscv = 1e3 # constraint violation
epsminimizer = 1e-2
epserrq = 1e-3
epsl = 1e-5
epsh = 1e-8
DEGTORAD = math.pi / 180
NOZZLE_ABS_ANGLE_MAX_WRT_GROUND = 90 * DEGTORAD # This is the angle measured w.r.t. ground frame
GFOLD_FRAME_INVALID = -1
GFOLD_FIXED_N = 200

# Global constants
g = 9.80665

GFOLD_INIT_POS_OFFSET = Vec3(0.0, 0.0, +0.05) # SET A BIT HIGHER such that the rocket doesn't stay still when recovery starts
rectified_z_target_factor = 0.90 # For the "feedforward" approach this factor should be lower, because by feeding forward the "z-velocity" would be very near 0 when near the landing spot, thus it might levitate a while and sour again if not coped with properly. 

def create_parameterized_guidanceprob(N, env):
    # TODO: Impose a "LOW CURVATURE EVERYWHERE" CONSTRAINT! See https://math.stackexchange.com/questions/1347848/curvature-of-a-3d-trajectory-for-which-i-know-data-points
    """
    Instead of using a fixed "gfold_dt", I tried to fix "N" for every search with varying "gfold_dt" for efficiency. 
    """
    parameterizedTofSeconds = opt.Parameter()
    gfold_dt = parameterizedTofSeconds/N # This value couldn't be too large, i.e. N shouldn't be too small to avoid unrealistic trajectory 

    nowPos, nowVel, nowLogm, nowQuat, nowOmega = env.state()
    rho1, rho2 = env.rho1, env.rho2
    m_wet = math.exp(nowLogm)

    descendingVelZLimit = 4.0
    descendingAccZLimit = 0.3*g # The "velZ" couldn't change too quickly 

    glide_tangent = math.tan(math.pi / 6) # Make this value SMALLER to allow WIDER cone

    logmLower = [(opt.log(m_wet - env.alpha * rho2 * i * gfold_dt)) for i in range(N)]
    mu1 = [(rho1 * opt.exp(-logmLower[i])) for i in range(N)]
    mu2 = [(rho2 * opt.exp(-logmLower[i])) for i in range(N)]

    r = opt.Variable((N, 6), 'r')  # state vector (3 position, 3 velocity)
    u = opt.Variable((N, 3), 'u')  # u = thrust/mass
    logm = opt.Variable(N, 'logm')  # logm = ln(mass)
    sigma = opt.Variable(N, 'sigma') # thrust slack parameter

    con = []  # CONSTRAINTS LIST
    con += [r[0, 0:3] == nowPos + GFOLD_INIT_POS_OFFSET] 
    con += [r[0, 3:6] == nowVel]  # initial velocity

    rectified_z_target = rectified_z_target_factor*env.physicsContainerNPRefPos[2]
    con += [ r[N-1, 0:3] == (env.physicsContainerNPRefPos[0], env.physicsContainerNPRefPos[1], rectified_z_target) ]  # end position, set the target a little under the COG at landing spot
    con += [ r[N-1, 3:6] == (env.physicsContainerNPRefVel[0], env.physicsContainerNPRefVel[1], env.physicsContainerNPRefVel[2]) ]  # end velocity
    con += [ sigma[N-1] == 0. ] # end thrust
    con += [ logm[N-1] <= logm[N-2] ] # Prevent fluctuating mass solution

    con += [logm[0] == nowLogm]  # initial mass

    for i in range(1, N-1):
        #con += [ r[2, i] >= -0.8 ] # Don't go underground

        con += [ logm[i+1] - logm[i-1] == - 2 * gfold_dt * env.alpha * sigma[i] ]
        con += [ logm[i] <= logm[i-1] ] # Prevent fluctuating mass solution
        con += [ 0 <= sigma[i] ]

        con += [ r[i+1, 3:6] - r[i-1, 3:6] == 2*gfold_dt*(u[i, 0:3] + numpy.array([0., 0., -g])) ] # regular velocity-acceleration constraint 
        con += [ r[i+1, 0:3] - r[i-1, 0:3] == 2*gfold_dt*(r[i, 3:6]) ] # regular position-velocity constraint

        con += [ r[i+1, 0:3] -2*r[i, 0:3] + r[i-1, 0:3] == gfold_dt*gfold_dt*(u[i, 0:3] + numpy.array([0., 0., -g])) ] # 2nd order position-acceleration constraint

        con += [opt.SOC(sigma[i], u[i, 0:3])] # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t. 

        con += [mu1[i] * (1 - (logm[i] - logmLower[i]) + 0.5*(logm[i] - logmLower[i])**2) <= sigma[i],
                sigma[i] <= mu2[i] * (1 - (logm[i] - logmLower[i]))]

        con += [opt.log(m_wet - env.alpha * rho2 * i * gfold_dt) <= logm[i], logm[i] <= opt.log(m_wet - env.alpha * rho1 * i * gfold_dt)]

        """
        [OPTIONAL, won't break SOCP compliance]
        "nozzle angle constraint" described at the end of the main paper as well as "Lossless Convexification of Nonconvex Control
Bound and Pointing Constraints of the Soft Landing Optimal Control Problem" by Behçet Açıkmeşe and John M. Carson III, and Lars Blackmore.
        """
        #con += [ sigma[i]*math.cos(NOZZLE_ABS_ANGLE_MAX_WRT_GROUND) <= u[i, 0:3]@numpy.array([0, 0, 1]) ]
        """
        [OPTIONAL, won't break SOCP compliance]
        Shouldn't descend too fast at any point of time 
        """
        #con += [ r[i-1, 2]-r[i+1, 2] <= 2*gfold_dt*descendingVelZLimit ]
        #con += [ opt.abs(r[i-1, 5]-r[i+1, 5]) <= 2*gfold_dt*descendingAccZLimit ]

        """
        [OPTIONAL, won't break SOCP compliance]
        Shouldn always be descending 
        """
        #con += [ r[i+1, 2] - r[i, 2] <= 0 ]

    objective = opt.Maximize(logm[N-1]) 
    problem = opt.Problem(objective, con)
    rootLogger.warning(f'problem types: is_dpp={problem.is_dpp()}, is_dqcp={problem.is_dqcp()}, is_dcp={problem.is_dcp()}, is_qp={problem.is_qp()}')
    return problem, parameterizedTofSeconds

def search_result(problem, parameterizedTofSeconds, timeOfFlightSeconds, foundNEvent, resultHolder, tf_min, tf_max):
    """
    [WARNING] Both variables "foundNEvent" & "resultHolder" are guarded by "gfoldMux"!!!
    """

    gfoldMux.acquire()
    if foundNEvent.isSet() is True:
        gfoldMux.release()
        return

    gfoldMux.release()

    rootLogger.info(f'Searching result for tf={timeOfFlightSeconds}')

    # [WARNING] The "problem" is SOCP and MUST be solvable for some N within [Nmin, Nmax], if not please check the convexity and SOCP compliance as well as the solver output by "verbose=True" carefully! 
    # For solver choices, see https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options
    parameterizedTofSeconds.value = timeOfFlightSeconds
    problem.solve(solver=opt.ECOS, verbose=False) 

    r, u, logm, sigma = None, None, None, None
    for var in problem.variables():
        if 'r' == var.name():    
            r = var
        elif 'u' == var.name():    
            u = var
        elif 'logm' == var.name():    
            logm = var
        elif 'sigma' == var.name():    
            sigma = var


    with gfoldMux:
        if foundNEvent.isSet() is True:
            gfoldMux.release()
            return

    if r.value is None:
        rootLogger.warning(f'Solution not found for timeOfFlightSeconds = {timeOfFlightSeconds:+05.2f} #3') 
        return

    rootLogger.info(f'Found timeOfFlightSeconds = {timeOfFlightSeconds:+05.2f} is feasible')
    with gfoldMux: 
        if 0 == len(resultHolder): 
            rootLogger.info(f'\tinitializing the resultHolder')
            resultHolder.append(timeOfFlightSeconds)
            resultHolder.append(r.value)
            resultHolder.append(u.value)
            resultHolder.append(logm.value)
            resultHolder.append(sigma.value)
        elif logm.value[-1] > resultHolder[3][-1]:
            rootLogger.info(f'\tand it\'s a better result than the previously found one')
            resultHolder[0] = (timeOfFlightSeconds)
            resultHolder[1] = (r.value)
            resultHolder[2] = (u.value)
            resultHolder[3] = (logm.value)
            resultHolder[4] = (sigma.value)
            
        if timeOfFlightSeconds >= tf_max-1:
            foundNEvent.set()

class Game(DirectObject):
    def __init__(self):
        base.enableParticles()
        base.setBackgroundColor(0.1, 0.1, 0.1, 1)
        base.setFrameRateMeter(True)

        self.nowEp = 0
        rootLogger.info(f'{args.ep} episodes to run')

        self.font = None
        #self.landerInitPos = Point3(+40, +40, 60)
        self.landerInitPos = Point3(+40, +80, 100)
        self.camOffset = Vec3(20.0, -20.0, 20.0)

        # By default x points to the right and y to forward, z to the up, see https://docs.panda3d.org/1.10/python/programming/scene-graph/common-state-changes 
        base.cam.setPos(self.landerInitPos + self.camOffset)
        base.cam.lookAt(self.landerInitPos)

        # Light
        alight = AmbientLight('ambientLight')
        alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        alightNP = render.attachNewNode(alight)

        dlight = DirectionalLight('directionalLight')
        dlight.setDirection(Vec3(1, 1, -1))
        dlight.setColor(Vec4(0.7, 0.7, 0.7, 1))
        dlightNP = render.attachNewNode(dlight)

        render.clearLight()
        render.setLight(alightNP)
        render.setLight(dlightNP)

        render.setAntialias(AntialiasAttrib.MAuto)

        # Physics
        '''
        [WARNING] 
        When "runPidEdCnt > doPhysicsCnt", the invocation to "lander.applyThrusts(dt)" will accumulate the "m_totalForce & m_totalTorque on the rigid body lander.physicsContainerNP.node()"! See https://github.com/bulletphysics/bullet3/blob/master/src/BulletDynamics/Dynamics/btRigidBody.h for the actual implementation.# [WARNING] When "runPidEdCnt > doPhysicsCnt", the invocation to "lander.applyThrusts(dt)" will accumulate the "m_totalForce & m_totalTorque on the rigid body lander.physicsContainerNP.node()"! See https://github.com/bulletphysics/bullet3/blob/master/src/BulletDynamics/Dynamics/btRigidBody.h for the actual implementation.
        Moreover, using "self.applyThrustsInMainThread = True" is risky when "UpdateWorld" runs more frequently than expected, i.e. in the main thread "lander.applyThrusts(dt)" is called many times for an OUTDATED CALC RESULT from "PidTask"!  
        '''
        taskMgr.setupTaskChain('ControllerTaskChain', numThreads=1, threadPriority=TP_urgent, tickClock=False, frameSync=True) # Reference https://docs.panda3d.org/1.10/python/programming/tasks-and-events/task-chains#task-chains 
        self.applyThrustsInMainThread = True # Will have impact on "doPhysics:runPidEdCnt", why?
        self.setup()
        # [WARNING] Sympy calculations should be only invoked ONCE FOR ALL EPISODES, because "sympy" has well-known memory leak issues even with SYMPY_USE_CACHE=no! Search in internet for more info. 
        self.torqueAndCogForceFromFmagList, self.torqueAndCogForceFromFmagListJac, self.torqueFromFmagList, self.torqueFromFmagListJac, self.cogForceFromFmagList, self.cogForceFromFmagListJac, self.fmagListCvx, self.targetTorqueAndCogForceInBodyParamsCvx, self.ctrlproblem, self.torqueAndCogForcePseudoInverse = self.lander.calcThrustorIKForControllerEstimation()
        self.prevTargetTorqueAndCogForceInBody = None
        self.runGFOLD()

        # [WARNING] The order of the following statements matters!
        self.elapsedMsArr = [None]*GFOLD_FIXED_N
        self.posArr = [None]*GFOLD_FIXED_N
        self.velArr = [None]*GFOLD_FIXED_N
        self.massArr = [None]*GFOLD_FIXED_N
        self.quatArr = [None]*GFOLD_FIXED_N
        self.globalDtArr = [None]*GFOLD_FIXED_N
        self.maxcvArr = [None]*GFOLD_FIXED_N
        self.phyDivPidStArr = [None]*GFOLD_FIXED_N
        self.phyDivPidEdArr = [None]*GFOLD_FIXED_N

        self.plannedPosArr = [None]*GFOLD_FIXED_N
        self.plannedVelArr = [None]*GFOLD_FIXED_N
        self.plannedMassArr = [None]*GFOLD_FIXED_N
        self.reset()
        taskMgr.add(self.update, 'UpdateWorld')
        taskMgr.add(self.runPid, 'PidTask', taskChain='ControllerTaskChain')
        self.runNewEpisode()
        self.acceptInputs()

    # _____HANDLER_____

    def doExit(self):
        self.cleanup()
        sys.exit(1)

    def reset(self):
        self.lander.reset()
        # Reset cvx problem to guarantee clearing cache

        self.runningTasks = 2
        self.shouldStartController = False
        self.isRecovering = False
        self.shouldEndCurrentEpisode = False # Both "runPid" and "update" will start proceeding from now on
        self.doPhysicsCnt = 0
        self.runPidStCnt = 1 # To avoid division by 0
        self.runPidEdCnt = 1 # To avoid division by 0
        self.recoveryStartTimeInMainThread = 0.
        self.elapsedTimeInMainThread = 0.
        rootLogger.warning(f'reset(): self.recoveryStartTimeInMainThread = {self.recoveryStartTimeInMainThread}, self.elapsedTimeInMainThread = {self.elapsedTimeInMainThread}')

        # Stats, use fixed size arrays to avoid inflight memory allocation while running the panda tasks
        for i in range(GFOLD_FIXED_N):
            self.elapsedMsArr[i] = None
            self.posArr[i] = None
            self.velArr[i] = None
            self.massArr[i] = None
            self.quatArr[i] = None
            self.globalDtArr[i] = None
            self.maxcvArr[i] = None
            self.phyDivPidStArr[i] = None
            self.phyDivPidEdArr[i] = None

            self.plannedPosArr[i] = None
            self.plannedVelArr[i] = None
            self.plannedMassArr[i] = None
        # End Stats

        self.pidStatePrint()
        base.cam.setPos(self.landerInitPos + self.camOffset)
        base.cam.lookAt(self.landerInitPos)

    def runNewEpisode(self):
        rootLogger.warning(f'Running new episode at self.nowEp = {self.nowEp}')

        # Just don't re-exec "runGFOLD"!
        self.triggerRecovery()
        
    def runGFOLD(self):
        env = self.lander

        nowPos, nowVel, nowLogm, nowQuat, nowOmega = env.state()

        tf_min = 13 # seconds
        tf_max = 20 # seconds

        rootLogger.info(f'tf_min: {tf_min}, tf_max: {tf_max}')
            
        resultHolder = []
        foundNEvent = Event()
        foundNEvent.clear()
        parameterizedProb, parameterizedTofSeconds = create_parameterized_guidanceprob(GFOLD_FIXED_N, env)
        for tf in range(tf_min, tf_max, 1):
            gevent.spawn(search_result, parameterizedProb, parameterizedTofSeconds, tf, foundNEvent, resultHolder, tf_min, tf_max)

        timeoutSeconds = 1200.0
        rootLogger.info(f'Starting to await resultHolder for {timeoutSeconds} seconds')
        foundNEvent.wait(timeout=timeoutSeconds)
        #if foundNEvent.isSet() is False:
        if 0 == len(resultHolder):
            rootLogger.error("No valid timeOfFlightSeconds is found!")
            sys.exit(0)
        
        timeOfFlightSeconds, r, u, logm, sigma = resultHolder

        rootLogger.info(f'timeOfFlightSeconds is {timeOfFlightSeconds:+05.2f}\nr[-1, 0:3] is {r[-1, 0:3]}\nr[-1, 3:6] is {r[-1, 3:6]}\nmass[-1] is {math.exp(logm[-1])}\ntarget is {self.lander.physicsContainerNPRefPos}')

        # avoid slicing during each tick of "runPid"
        self.planned_pos_trajectory = r[:, 0:3] 
        stringifiedPosTrajectory = '\n'.join([str(p) for p in self.planned_pos_trajectory]) 
        rootLogger.info(f'The whole trajectory:\n{stringifiedPosTrajectory}')

        self.planned_vel_trajectory = r[:, 3:6]
        self.planned_logm_trajectory = logm
        self.kd_tree_planned_trajectory = KDTree(self.planned_pos_trajectory) 
        self.timeOfFlightSeconds = timeOfFlightSeconds
        self.gfoldN = GFOLD_FIXED_N # don't assume that "self.gfoldN" being always constant, it's merely a local convenience of the current approach
        self.gfoldDt = (timeOfFlightSeconds/self.gfoldN)

    def acceptInputs(self):
        # Only initialize once, otherwise will cause memoryleak! 
        self.accept('escape', self.doExit)
        self.accept('f1', self.toggleWireframe)
        self.accept('f2', self.toggleTexture)
        self.accept('f3', self.toggleDebug)
        self.accept('f5', self.doScreenshot)
        self.accept('b', self.bodyeigenprint)
        self.accept('o', self.runNewEpisode)
        self.accept('p', self.triggerEnd)

        inputState.watchWithModifiers('yaw-', 'a')
        inputState.watchWithModifiers('yaw+', 'd')
        inputState.watchWithModifiers('pitch-', 's')
        inputState.watchWithModifiers('pitch+', 'w')
        inputState.watchWithModifiers('roll-', 'q')
        inputState.watchWithModifiers('roll+', 'e')

    def toggleWireframe(self):
        base.toggleWireframe()

    def toggleTexture(self):
        base.toggleTexture()

    def toggleDebug(self):
        if self.debugNP.isHidden():
            self.debugNP.show()
        else:
            self.debugNP.hide()

    def doScreenshot(self):
        base.screenshot('Bullet')

    def bodyeigenprint(self):
        angle, direction = self.lander.angleAxisOfBody()
        nowQuat = self.lander.physicsContainerNP.getQuat()
        angleFromQuat = nowQuat.getAngleRad()
        dirFromQuat = nowQuat.getAxisNormalized()

        # See https://pybullet.org/Bullet/BulletFull/btRigidBody_8cpp_source.html, "btRigidBody::computeGyroscopicImpulseImplicit_World" for a reference
        effectiveImatrixBody = self.lander.effective_ImatrixBody()
        rootLogger.info(f'nowPos is {self.lander.physicsContainerNP.getPos()}\nnowQuat is {nowQuat}\neigendir is {direction} with theta={angle}, to check dirFromQuat={dirFromQuat}, angleFromQuat={angleFromQuat},\neffectiveImatrixBody=\n{effectiveImatrixBody}')

    def pidStatePrint(self):
        # Draw onscreen text
        osdText = self.lander.plantDataArr()
        osdText.append('bynow (s)     :  ' + f'{self.elapsedTimeInMainThread:+6.0f}')
        osdText.append('WindVel (m/s) : (' + ','.join([f'{self.wind.vel(self.lander)[i]:+04.1f}' for i in range(3)]) + ')')
        osdText.append('doPhysicsCnt  :  ' + str(self.doPhysicsCnt))
        osdText.append('phyCnt/pidStCnt      :  ' + str(self.doPhysicsCnt/self.runPidStCnt))
        osdText.append('phyCnt/pidEdCnt      :  ' + str(self.doPhysicsCnt/self.runPidEdCnt))

        self.osdNode.setText('\n'.join(osdText))

    def triggerRecovery(self):
        with self.mux:
            self.shouldStartController = True

    def _startRecovery(self):
        self.isRecovering = True
        self.shouldStartController = False
        self.recoveryStartTimeInMainThread = self.elapsedTimeInMainThread 
        rootLogger.info(f'isRecovering = {self.isRecovering}, self.recoveryStartTimeInMainThread = {self.recoveryStartTimeInMainThread}, self.elapsedTimeInMainThread = {self.elapsedTimeInMainThread}')

    # ____TASK___
    def processManualRotation(self, dt, nowPos, nowQuat):
        torque = Vec3(0, 0, 0)

        if inputState.isSet('yaw-'):   torque.setZ(-5.0)
        if inputState.isSet('yaw+'):   torque.setZ(+5.0)
        if inputState.isSet('pitch-'): torque.setY(-5.0)
        if inputState.isSet('pitch+'): torque.setY(+5.0)
        if inputState.isSet('roll-'):  torque.setX(-5.0)
        if inputState.isSet('roll+'):  torque.setX(+5.0)
        torque *= 2.0
        if torque[0] != 0 or torque[1] != 0 or torque[2] != 0:
            self.lander.physicsContainerNP.node().applyTorque(torque)  # world frame

    def update(self, task):
        globalDt = globalClock.getDt()
        maxcv = None
        self.elapsedTimeInMainThread += globalDt
        nowPos, nowVel, nowLogm, nowQuat, nowOmega = None, None, None, None, None

        # Just initializing "planned_r_now" and "planned_r_dot_now" to the final value, and they should be updated with protection from the mutex "self.mux" 
        gfoldFrameCnt = GFOLD_FRAME_INVALID
        planned_r_now = self.planned_pos_trajectory[self.gfoldN-1]
        planned_r_dot_now = self.planned_vel_trajectory[self.gfoldN-1]
        planned_mass_now = math.exp(self.planned_logm_trajectory[self.gfoldN-1])

        localShouldEnd = False # Set for escaping the scope of "with self.mux"
        with self.mux:
            if self.shouldStartController is True and self.isRecovering is False:
                self._startRecovery() # Ensures that in every episode, "runPid" is started after "update"
            if self.shouldEndCurrentEpisode is True:
                localShouldEnd = True 
            else:
                nowPos, nowVel, nowLogm, nowQuat, nowOmega = self.lander.state()
                if self.isRecovering is True:
                    self.wind.applyAeroForceToFaces(self.lander) 
                    if self.applyThrustsInMainThread is True:
                        self.lander.applyThrusts(globalDt) 
                    # [WARNING] In fact, I want to call "self.world.doPhysics" even if "self.isRecovering is False", but experiment shows that such change cracks stability of lander attitude while landing -- root cause to be investigated  
                    self.world.doPhysics(globalDt, self.maxSubSteps, self.simulationFixedDt)
                    self.doPhysicsCnt += 1 # Only count when recovering
                    gfoldFrameCnt = self.lander.gfoldFrameCnt
                    maxcv = self.lander.minimizerMaxcv

        if localShouldEnd is True:
            rootLogger.warning(f'Calling tryEnd() from update')
            self.tryEnd()
            return task.cont

        if GFOLD_FRAME_INVALID != gfoldFrameCnt and gfoldFrameCnt < self.gfoldN:
            planned_r_now = self.planned_pos_trajectory[gfoldFrameCnt]
            planned_r_dot_now = self.planned_vel_trajectory[gfoldFrameCnt]     
            planned_mass_now = math.exp(self.planned_logm_trajectory[gfoldFrameCnt])     
            self.elapsedMsArr[gfoldFrameCnt] = (self.elapsedTimeInMainThread*1000.0)
            self.posArr[gfoldFrameCnt] = (nowPos)
            self.velArr[gfoldFrameCnt] = (nowVel)
            self.massArr[gfoldFrameCnt] = (math.exp(nowLogm))
            self.quatArr[gfoldFrameCnt] = (nowQuat)
            self.globalDtArr[gfoldFrameCnt] = (globalDt)
            self.maxcvArr[gfoldFrameCnt] = (maxcv)
            self.plannedPosArr[gfoldFrameCnt] = (planned_r_now)
            self.plannedVelArr[gfoldFrameCnt] = (planned_r_dot_now)
            self.plannedMassArr[gfoldFrameCnt] = (planned_mass_now)
            self.phyDivPidStArr[gfoldFrameCnt] = (self.doPhysicsCnt/self.runPidStCnt)
            self.phyDivPidEdArr[gfoldFrameCnt] = (self.doPhysicsCnt/self.runPidEdCnt)

        self.pidStatePrint()

        inContactLegs = set()
        result = self.world.contactTest(self.terrain.terrainRigidBodyNP.node())
        if 0 < result.getNumContacts():
            for item in result.getContacts():
                if item.node0.name.startswith('Leg'):
                    inContactLegs.add(item.node0.name)
                if item.node1.name.startswith('Leg'):
                    inContactLegs.add(item.node1.name) 

        base.cam.setPos(nowPos + self.camOffset)
        base.cam.lookAt(self.lander.physicsContainerNP)

        if self.isRecovering is True and 4 == len(inContactLegs):
            rootLogger.info(f'All 4 legs are in contact!')
            with self.mux:
                self.shouldEndCurrentEpisode = True # Would end this task at next tick 

        return task.cont

    def runPid(self, task): 
        globalDt = globalClock.getDt()

        nowPos, nowLinearVel, nowLogm, nowQuat, omegaInWorld = None, None, None, None, None
        nowMass = None
        gfold_dt = None
        errPos, accErrPos, errPosDt = None, None, None
        errQuat, accErrQuat, errQuatDt = None, None, None

        localShouldEnd = False
        with self.mux: 
            if self.shouldEndCurrentEpisode is True:
                localShouldEnd = True
            else:
                if self.isRecovering is False:
                    return task.cont

                nowPos, nowLinearVel, nowLogm, nowQuat, omegaInWorld = self.lander.state() # the "state()" method itself is NOT THREAD-SAFE!

        if localShouldEnd is True:
            rootLogger.warning(f'Calling tryEnd() from runPid')
            self.tryEnd()
            return task.cont

        self.runPidStCnt = self.runPidStCnt+1  
        calcCheckpt1 = time.time()
        gfold_dt = self.gfoldDt    
        nowMass = math.exp(nowLogm) 

        """
        Always aiming at the NEXT MILESTONE "idx+1" is wrong, consider the following cases

        case #0
        ---------------------> sequence direction 
        nowPos [idx] [idx+1]           [idx+2]
        : in {ii}, the order would be [idx, idx+1, idx+2] or [idx, idx-1, idx-2]

        case #1
        ---------------------> sequence direction 
        [idx-1] [idx] nowPos          [idx+1]
        : in {ii}, the order would be [idx, idx-1, idx+1]

        case #2
        ---------------------> sequence direction 
        [idx-1]         nowPos [idx] [idx+1]
        : in {ii}, the order would be [idx, idx+1, idx-1]

        case #3
        ---------------------> sequence direction 
        [idx-1]  nowPos[idx]         [idx+1]
        : in {ii}, the order would be [idx, idx-1, idx+1]

        case #4
        ---------------------> sequence direction 
        [idx-1]         [idx] nowPos [idx+1]
        : in {ii}, the order would be [idx, idx+1, idx-1]


        In case#1 we should choose "gfoldFrameCnt = idx+1"; in both case#2 & case#3 "gfoldFrameCnt = idx" instead; in case#4 should choose "gfoldFrameCnt = idx+1".
        """
        _, ii = self.kd_tree_planned_trajectory.query(x=nowPos, k=3)
        caseChoice = None
        pPlus1 = self.planned_pos_trajectory[ii[2]]
        pMinus1 = self.planned_pos_trajectory[ii[0]]
        vPlus1 = Vec3(pPlus1[0], pPlus1[1], pPlus1[2]) - nowPos
        vMinus1 = Vec3(pMinus1[0], pMinus1[1], pMinus1[2]) - nowPos
        dotProd = vPlus1.dot(vMinus1)  
        if (ii[0]-ii[1]) * (ii[0]-ii[2]) > 0:
            # case#0
            caseChoice = 0
            gfoldFrameCnt = ii[0]   
        else:
            if ii[1] < ii[2]:
                # Check case#1 or case#3
                if dotProd < 0: 
                    # case#1
                    caseChoice = 1
                    gfoldFrameCnt = ii[2]   
                else:
                    # case#3
                    caseChoice = 3
                    gfoldFrameCnt = ii[0]   
            else:
                # Check case#2 or case#4
                if dotProd < 0:
                    # case#4
                    caseChoice = 4
                    gfoldFrameCnt = ii[2]   
                else:
                    # case#2
                    caseChoice = 2
                    gfoldFrameCnt = ii[0]   

        #rootLogger.info(f'ii is {ii}, caseChoice is {caseChoice}, gfoldFrameCnt is {gfoldFrameCnt}')
        if gfoldFrameCnt < self.lander.gfoldFrameCnt:
            #rootLogger.warning(f'FIXME!!!!! gfoldFrameCnt order reversed! ii is {ii}, caseChoice is {caseChoice}, gfoldFrameCnt is {gfoldFrameCnt} for pPlus1 = {pPlus1}, pMinus1 = {pMinus1}, nowPos = {nowPos}')
            gfoldFrameCnt = self.lander.gfoldFrameCnt

        #rootLogger.info(f'gfoldFrameCnt is {gfoldFrameCnt} for pPlus1 = {pPlus1}, pMinus1 = {pMinus1}, nowPos = {nowPos}')

        calcCheckpt2 = time.time()
         
        planned_quat_now = self.lander.physicsContainerNPRefQuat
        planned_r_now = self.planned_pos_trajectory[self.gfoldN-1]
        planned_r_dot_now = self.planned_vel_trajectory[self.gfoldN-1]
        planned_r_dot_dot_now = Vec3(0., 0., 0.)

        if gfoldFrameCnt < self.gfoldN:
            planned_r_now = self.planned_pos_trajectory[gfoldFrameCnt]
            planned_r_dot_now = self.planned_vel_trajectory[gfoldFrameCnt]     
            planned_r_dot_dot_now_arr = numpy.array((planned_r_dot_now - self.planned_vel_trajectory[gfoldFrameCnt-1]) if (gfoldFrameCnt-1 >= 0) else ([0., 0., 0.]))/gfold_dt 
            planned_r_dot_dot_now = Vec3(planned_r_dot_dot_now_arr[0], planned_r_dot_dot_now_arr[1], planned_r_dot_dot_now_arr[2]) 

        # For actually recovering the orientation, quaternion is the best fit here because it was invented for this purpose, i.e. a compact representation of "Euler's rotation theorem".
        invNowQuat = nowQuat.conjugate()  # for unit quaternion the conjugate equals the reciprocal(a.k.a. inverse)
        # "inverse(nowQuat)*planned_quat_now" reads "difference from nowQuat to planned_quat_now"
        errQuat = invNowQuat * planned_quat_now
        omegaQuat = Quat(0.0, omegaInWorld.getX(), omegaInWorld.getY(), omegaInWorld.getZ())
        qDot = omegaQuat*nowQuat*0.50 # By the knowledge that "dQ(t)/dt = 0.5*W(t)*Q(t)" in reference PDF "QuaternionDifferentiation" 
        errQuatDt = qDot*-1.0
    
        errPos = Point3(planned_r_now[0], planned_r_now[1], planned_r_now[2]) - nowPos
        errPosDt = Vec3(planned_r_dot_now[0], planned_r_dot_now[1], planned_r_dot_now[2]) - nowLinearVel

        accErrPos = self.lander.accErrPos + errPos
        accErrQuat = self.lander.accErrQuat + errQuat # FIXME: Anyother thought?
        self.lander.accErrPos = accErrPos 
        self.lander.accErrQuat = accErrQuat 

        """
        # To verify that the "errQuat" is correctly calculated in world coordinate system
        errHpr = errQuat.getHpr() 
        nextHpr = self.lander.physicsContainerNP.getHpr()
        nextHpr += errHpr
        self.lander.physicsContainerNP.setHpr(nextHpr)
        """

        # Turn this "errQuat" to "torqueWorld"
        dzTot = abs(self.lander.initialPos[2] - self.lander.physicsContainerNPRefPos[2])
        dz = abs(nowPos[2] - self.lander.physicsContainerNPRefPos[2])
        KpQuatFactor = 1-(dz/dzTot) # Only gets stronger quat feedback when closing to target in altitude 
        KpQuat = 3.0*KpQuatFactor 
        KiQuat = 0.000
        KdQuat = 5.0

        KpLinear = 3.0
        KiLinear = 0.001
        KdLinear = 5.0
        # Calculate world frame set-points of accelerations
        targetQuatDotDotInWorld = (errQuat * KpQuat + accErrQuat * KiQuat + errQuatDt * KdQuat)  
        targetBetaQuatInWorld = (targetQuatDotDotInWorld*invNowQuat - (qDot*invNowQuat)*(qDot*invNowQuat))*2.0 # see "QuaternionDifferentiation.pdf" 
        targetBetaInWorld = Vec3(targetBetaQuatInWorld.getI(), targetBetaQuatInWorld.getJ(), targetBetaQuatInWorld.getK())   
        
        targetCogAccInWorldFeedforwardTerm = (planned_r_dot_dot_now)
        targetCogAccInWorld = (errPos * KpLinear + accErrPos * KiLinear + errPosDt * KdLinear) + targetCogAccInWorldFeedforwardTerm      

        targetCogNetEngineForceInWorld = (targetCogAccInWorld - self.world.getGravity())*nowMass   
        # Calculate body frame set-points of accelerations
        
        # It's easier to calculate "targetTorque" in body frame because there's an easier expression of "inertia tensor".
        targetBetaInBody = invNowQuat.xform(targetBetaInWorld)
        omegaInBody = invNowQuat.xform(omegaInWorld)
        effectiveImatrixBody = self.lander.effective_ImatrixBody()  
        targetTorqueInBody = effectiveImatrixBody.xform(targetBetaInBody) + omegaInBody.cross(effectiveImatrixBody.xform(omegaInBody))
        targetCogNetEngineForceInBody = invNowQuat.xform(targetCogNetEngineForceInWorld)   

        targetTorqueAndCogForceInBody = numpy.array([
                targetTorqueInBody[0], 
                targetTorqueInBody[1], 
                targetTorqueInBody[2],
                targetCogNetEngineForceInBody[0], 
                targetCogNetEngineForceInBody[1], 
                targetCogNetEngineForceInBody[2] 
                #numpy.clip(targetCogNetEngineForceInBody[2], 0., None)
        ]) # Note that we couldn't "thrust down in body frame", i.e. there's no such engine, but clipping will be done anyway at last -- thus keep the targetCogNetEngineForceInBodyArr here as-is  

        scaleFactor = (1.0/(nowMass*10.0))

        def cmpslsqp2(fmagList):
            torqueAndCogForceResArr = self.torqueAndCogForceFromFmagList(fmagList);
            return (torqueAndCogForceResArr - targetTorqueAndCogForceInBody)

        #initGuess = numpy.array([epsh, epsh, epsh, epsh, epsh, epsh, epsh, epsh])  
        quaterLowerThrustGuess = nowMass*(-self.world.getGravity().getZ())*0.25
        initGuess = numpy.array([epsh, epsh, epsh, epsh, quaterLowerThrustGuess, quaterLowerThrustGuess, quaterLowerThrustGuess, quaterLowerThrustGuess]) # Start with an only-gravity-balanced guess 
        #initGuess = self.lander.engineFmagList # Use previous solution
        #initGuess = self.torqueAndCogForcePseudoInverse@targetTorqueAndCogForceInBody

        # Best choice is SLSQP_INPUT_SCALED by far
        methodChoice = 'SLSQP_INPUT_SCALED' # The use of 'linprog' is not yet considered because the "torqueAndCogForceFromFmagList" will be nonlinear w.r.t. "fmagList" once any thruster is to be gimbaled.
        #methodChoice = 'CVX_INPUT_SCALED' 

        sol = initGuess
        solMin = None
        minimizerMaxcv = 0.0
        shouldPrioritizeAttitude = False # Default to false
        calcCheckpt21 = time.time()
        if 'SLSQP_INPUT_SCALED' == methodChoice: 
            sol, solMin = self.solveBySLSQP(initGuess, targetTorqueAndCogForceInBody, scaleFactor, True)
            #rootLogger.info(f'solMin is {solMin}')
        elif 'SLSQP' == methodChoice: 
            sol, solMin = self.solveBySLSQP(initGuess, targetTorqueAndCogForceInBody, scaleFactor, False)
            #rootLogger.info(f'solMin is {solMin}')
        elif 'CVX_INPUT_SCALED' == methodChoice:
            sol = self.solveByCvx(initGuess, targetTorqueAndCogForceInBody, scaleFactor, True, True)
        elif 'CVX' == methodChoice:
            sol = self.solveByCvx(initGuess, targetTorqueAndCogForceInBody, scaleFactor, False, True)
        else:
            raise Exception('No solver chosen!')

        calcCheckpt3 = time.time()
        
        self.prevTargetTorqueAndCogForceInBody = targetTorqueAndCogForceInBody

        sol = self.lander.clipThrust(sol, 0.05*self.lander.rho1, self.lander.rho2, False)
        minimizerMaxcv = numpy.linalg.norm(cmpslsqp2(sol))
        if solMin is not None and (minimizerMaxcv > epscv):
            rootLogger.error(f'globalDt = {globalDt}; Failed to solve SLSQP for gfoldFrameCnt={gfoldFrameCnt}\nminimizerMaxcv={minimizerMaxcv}\n\n')            
            pass

        # Apply the computed forces immediately, kindly note that we do it from the controller thread, because the main physics simulation thread is NOT interruptable in reality!
        fmagList = sol
            
        with self.mux:
            self.lander.applyThrustsDryrun(fmagList, gfoldFrameCnt)
            self.lander.minimizerMaxcv = minimizerMaxcv
            if self.applyThrustsInMainThread is False:
                self.lander.applyThrusts(globalDt) 

        calcCheckpt4 = time.time()

        self.runPidEdCnt = self.runPidEdCnt+1 
        
        solverIsSlow = ((calcCheckpt3-calcCheckpt21) > 1/DESIRED_GUI_FPS)
        if solverIsSlow is True: 
            # [WARNING] A significant drop in minimizer performance will result in an increment of "doPhysicsCnt/runPidEdCnt", which in turn results in "self.update()" calling "lander.applyThrusts(dt)" too many times with an OUTDATED CALC RESULT!  
            #rootLogger.warning(f'calcCheckpt2-calcCheckpt1 = {calcCheckpt2-calcCheckpt1}\ncalcCheckpt21-calcCheckpt2 = {calcCheckpt21-calcCheckpt2}\ncalcCheckpt3-calcCheckpt21 = {calcCheckpt3-calcCheckpt21}\ncalcCheckpt4-calcCheckpt3 = {calcCheckpt4-calcCheckpt3}')
            rootLogger.warning(f'calcCheckpt3-calcCheckpt21 = {calcCheckpt3-calcCheckpt21}')
            pass

        return task.cont

    def triggerEnd(self):
        with self.mux:
            self.shouldEndCurrentEpisode = True        
        
    def tryEnd(self):
        localShouldEnd = False
        with self.mux:
            self.isRecovering = False # Should cut recovering just at "tryEnd" phase
            self.runningTasks -= 1
            rootLogger.warning(f'tryEnd(): #runningTasks {self.runningTasks}')
            if self.runningTasks == 0:
                localShouldEnd = True
        
        if localShouldEnd is True:
            self.doEnd()

    def doEnd(self):
        rootLogger.warning(f'doEnd()')

        # persist episode result
        filepath = CWD + f'/episode_stats/ep_{self.nowEp:02d}.stats'
        with open(filepath, 'wb') as statsFile: 
            rootLogger.info(f'Writting to {filepath}, please don\'t close the application!')
            measuredTimeOfFlightSeconds = self.elapsedTimeInMainThread - self.recoveryStartTimeInMainThread
            if 0. >= measuredTimeOfFlightSeconds:
                rootLogger.error(f'measuredTimeOfFlightSeconds is 0 when self.elapsedTimeInMainThread = {self.elapsedTimeInMainThread} and self.recoveryStartTimeInMainThread = {self.recoveryStartTimeInMainThread}, why?!')
                 
            todump = (measuredTimeOfFlightSeconds, self.timeOfFlightSeconds, self.elapsedMsArr, self.posArr, self.plannedPosArr, self.velArr, self.plannedVelArr, self.massArr, self.plannedMassArr, self.quatArr, self.globalDtArr, self.maxcvArr)
            pickle.dump(todump, statsFile)

        rootLogger.info(f'Written to {filepath}')

        with self.mux:
            self.reset()

        gc.collect()
        
        if args.debugleak > 0:
            # Don't print diffed objs from 0th to 1st, that'll be a lot
            all_objects = muppy.get_objects()
            #dump_objects = [o for o in all_objects if (isinstance(o, str) or isinstance(o, list) or isinstance(o, tuple) or isinstance(o, int))]
            dump_objects = [o for o in all_objects if (isinstance(o, str))]
            filepath = CWD + f'/heapdumps/ep_{self.nowEp}.hprof'
            with open(filepath, 'wb') as hdfile:
                rootLogger.info(f'Writting heapdump to {filepath}, please don\'t close the application!')
                pickle.dump(dump_objects, hdfile)

            rootLogger.info(f'Written heapdump to {filepath}')
            #summaryOfEpisode = summary.summarize(all_objects)
            #summary.print_(summaryOfEpisode)
            smytracker.print_diff()

        self.nowEp += 1
        
        if self.nowEp < args.ep:
            self.runNewEpisode()

    def cleanup(self):
        '''
        [WARNING]

        This "cleanup" is NOT WORKING AS EXPECTED. After some trial & error, there's still leak in the "C extension" part, most possibly within pybullet module, because a "reset" without deallocating the "BulletWorld" makes no leak in contrast. 
        '''
        # Panda3D NodePath variables don't need manual deallocation, they'll be deallocated once detached from "worldNP"
        taskMgr.remove("PidTask")
        taskMgr.remove("UpdateWorld")
        self.wind.cleanup()
        self.wind = None
        self.terrain.cleanup()
        self.terrain = None
        self.lander.cleanup()
        self.lander = None
        self.osdNP.removeNode() # Special NodePath which isn't attached to "self.worldNP"
        self.osdNode = None
        self.world = None
        self.worldNP.removeNode()

        self.elapsedMsArr = None
        self.posArr = None
        self.velArr = None
        self.massArr = None
        self.quatArr = None
        self.globalDtArr = None
        self.maxcvArr = None
        self.phyDivPidStArr = None
        self.phyDivPidEdArr = None

        self.plannedPosArr = None
        self.plannedVelArr = None
        self.plannedMassArr = None
        gc.collect()

    def setup(self):
        # Mutex lock for accessing shared data
        self.mux = threading.Lock() # The use of "self.mux" per tick in 2 threads will impact GUI performance significantly, thus not used yet.
        self.isRecovering = False  # RW guarded by "self.mux"
        self.shouldEndCurrentEpisode = False # RW guarded by "self.mux"
        self.runningTasks = 2 # RW guarded by "self.mux"

        self.worldNP = render.attachNewNode('World')

        # World
        self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
        self.debugNP.node().showWireframe(True)
        self.debugNP.node().showConstraints(True)
        self.debugNP.node().showBoundingBoxes(False)
        self.debugNP.node().showNormals(True)

        self.recoveryStartTimeInMainThread = 0.0
        self.elapsedTimeInMainThread = 0.0
        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -g))
        self.world.setDebugNode(self.debugNP.node())

        self.maxSubSteps = 1
        self.simulationFixedDt = 1.0/60 # the denominator is fps
    
        # Wind
        self.wind = Wind(1)

        # Terrain
        self.terrain = Terrain(self.world, self.worldNP)

        landerCOGTargetXY = [-40.0, -10.0] # Somewhere I observed to be flat in the GeoMipTerrain
        landerCOGOffsetZ = +0.8 # Estimated
        landerCOGTargetZ = self.terrain.calcRectifiedZ(landerCOGTargetXY[0], landerCOGTargetXY[1])

        landerCOGTargetZ += landerCOGOffsetZ
        self.landerTargetPos = Point3(landerCOGTargetXY[0], landerCOGTargetXY[1], landerCOGTargetZ)
        rootLogger.info(f'landerTargetPos = {self.landerTargetPos}')

        # Load target marker
        markerNP = loader.loadModel('models/flag.bam')
        markerNP.setScale(1.0)
        markerNP.setCollideMask(BitMask32.allOff())
        markerNP.setPos(self.landerTargetPos)
        markerNP.reparentTo(self.worldNP)

        # Lander (dynamic)
        self.lander = LunarLanderFuel(withNozzleImpluseToCOG=False, withNozzleAngleConstraint=False, withFuelTank=True) 
            
        self.lander.setup(self.world, self.worldNP, 2375.60, self.landerInitPos, self.landerTargetPos, BitMask32(0x10), 9.80665)
        self.doPhysicsCnt = 0
        self.runPidStCnt = 0 
        self.runPidEdCnt = 0

        # Onscreen TextNode, use white foreground color and transparent background color
        font = None
        if self.font is None:
            font = loader.loadFont('Cascadia.ttf') # Monowidth font   
            self.font = font
        else:
            font = self.font

        self.osdNode = TextNode('OnscreenText')
        self.osdNode.setFont(font)
        self.osdNode.setText('Awaiting start...............')
        self.osdNode.setTextColor(0.0, 0.0, 0.0, 1.0)
        self.osdNode.setCardColor(1, 1, 0.5, 1)
        self.osdNode.setCardAsMargin(0, 0, 0, 0)
        self.osdNode.setCardDecal(True)
        self.osdNode.setAlign(TextNode.ALeft)
        self.osdNP = base.a2dTopLeft.attachNewNode(self.osdNode)
        self.osdNP.setScale(0.05)
        self.osdNP.setPos(Point3(0.0, 0.0, -0.05)) # Only x-axis and z-axis are meaningful and the range should be [-1.0, +1.0]
    
    def solveBySLSQP(self, initGuess, targetTorqueAndCogForceInBody, scaleFactor, useInputScale):
        if useInputScale is True:
            def cmpslsqp2InputScaled(fmagList):
                return (self.torqueAndCogForceFromFmagList(fmagList) - targetTorqueAndCogForceInBody*scaleFactor)

            eq_cons = {
                'type': 'eq',
                'fun' : cmpslsqp2InputScaled,
            }
            solMin = minimize(
                fun = ( lambda fmagList: numpy.linalg.norm(cmpslsqp2InputScaled(fmagList)) ),  
                x0 = (initGuess*scaleFactor), 
                method='SLSQP',
                bounds=[(0.0, None)]*len(self.lander.enginePosList), 
                constraints=[eq_cons], # Interestingly, having the "eq_cons" present helps SLSQP obtain better solution here 
                options={'maxiter': 100, 'disp': False},             
            )
            return (solMin.x/scaleFactor), solMin
        else:
            def cmpslsqp2OutputScaled(fmagList):
                return (self.torqueAndCogForceFromFmagList(fmagList*scaleFactor) - targetTorqueAndCogForceInBody*scaleFactor)
            '''
            [TODO] If not approaching the landing site too closely, can we allow a bigger "errQuatMag" during the flight and only recover it to (1, 0, 0, 0) when it's certainly closing to the end?  
            '''
            errQuatMag = ((errQuat.getR()-1.0)**2 + errQuat.getI()**2 + errQuat.getJ()**2 + errQuat.getK()**2)
            # TODO: This "attitude recovery only" doesn't work by far, it even DETERIORATES the situation!
            #shouldPrioritizeAttitude = (errQuatMag > epserrq) # or maybe (minimizerMaxcv > epscv) after the first minimization process?  
            eq_cons = {
                'type': 'eq',
                'fun' : cmpslsqp2OutputScaled, 
            }
            # [WARNING] The system governed by "cmpslsqp2" is OVERDETERMINED, hence shouldn't be used as an equality constraint!
            solMin = minimize(
                fun = ( lambda fmagList: numpy.linalg.norm(cmpslsqp2OutputScaled(fmagList)) ),  
                x0 = initGuess, 
                method='SLSQP',
                bounds=[(0.0, None)]*len(self.lander.enginePosList), 
                constraints=[eq_cons], # Interestingly, having the "eq_cons" present helps SLSQP obtain better solution here 
                options={'maxiter': 100, 'disp': False},             
            )
            return solMin.x, solMin

    def solveByCvx(self, initGuess, targetTorqueAndCogForceInBody, scaleFactor, useInputScale, useSLSQPBackup):
        if useInputScale is True:
            self.targetTorqueAndCogForceInBodyParamsCvx.value = targetTorqueAndCogForceInBody*scaleFactor 
            #self.ctrlproblem.solve(solver=opt.SCS, verbose=False) 
            self.ctrlproblem.solve(solver=opt.ECOS, verbose=False) 
            if self.ctrlproblem.status not in ["infeasible", "unbounded"]:
                return [var.value/scaleFactor for i,var in enumerate(self.fmagListCvx)]
            else:
                rootLogger.warning(f'ctrlproblem.status: {self.ctrlproblem.status}\n\tsolver stats: {self.ctrlproblem.solver_stats.extra_stats}\ntargetTorqueAndCogForceInBody={targetTorqueAndCogForceInBody}\nprevTargetTorqueAndCogForceInBody={self.prevTargetTorqueAndCogForceInBody}')
        else:
            self.targetTorqueAndCogForceInBodyParamsCvx.value = targetTorqueAndCogForceInBody 
            self.ctrlproblem.solve(solver=opt.SCS, verbose=False) 
            #self.ctrlproblem.solve(solver=opt.ECOS, verbose=False) 
            if self.ctrlproblem.status not in ["infeasible", "unbounded"]:
                return [var.value for i,var in enumerate(self.fmagListCvx)]
            else:
                rootLogger.warning(f'ctrlproblem.status: {self.ctrlproblem.status}\n\tsolver stats: {self.ctrlproblem.solver_stats.extra_stats}\ntargetTorqueAndCogForceInBody={targetTorqueAndCogForceInBody}\nprevTargetTorqueAndCogForceInBody={self.prevTargetTorqueAndCogForceInBody}')
            
        if useSLSQPBackup is True:
            sol,_ = self.solveBySLSQP(initGuess, targetTorqueAndCogForceInBody, scaleFactor, useInputScale)
            return sol
        else:
            return initGuess

parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--ep', default=1, type=int,
                    help='number of episodes to run (default: 1)')
parser.add_argument('--debugleak', default=0, type=int,
                    help='whether or not to debug memoryleak (default: 0)')

args = parser.parse_args()
gc.disable() # Disable automatic gc to avoid impacting performance of SLSQP solver, reference https://docs.python.org/3/library/gc.html
game = Game()
base.run()
