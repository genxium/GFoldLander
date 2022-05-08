# -*- coding: utf-8 -*-

from gevent import monkey
monkey.patch_all()

import colorlog as logging
import os
import sys
import bisect
from cloudpickle import dumps, loads

logFormatter = logging.ColoredFormatter("%(log_color)s %(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]\n%(message)s")

rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.DEBUG)
rootLogger.propagate = False # disables console logging if not later explicitly added

from gym.wrappers.monitoring import video_recorder

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

CWD = os.path.dirname(os.path.abspath(__file__))

rootLogger.info('CWD=%s', CWD)

sys.path.append(CWD)

from control_and_ai import pd_designer
from environments.lunarlanderfuel import LunarLanderFuel
import numpy as np
import math, time
import cvxpy as opt
from control_and_ai.pid import PID
import matplotlib.pyplot as plt
from sympy import Symbol, lambdify

import gevent
from gevent.event import Event
from gevent.lock import BoundedSemaphore
mux = BoundedSemaphore(1)

rectified_y_target_factor = 0.95 # For the "feedforward" approach this factor should be lower, because by feeding forward the "y-velocity" would be very near 0 when near the landing spot, thus it might levitate a while and sour again if not coped with properly. 

def search_result(N, env, foundNEvent, resultHolder):
    """
    [WARNING] Both variables "foundNEvent" & "resultHolder" are guarded by "mux"!!!
    """

    mux.acquire()
    if foundNEvent.isSet() is True:
        mux.release()
        return

    mux.release()

    posNow, velNow, zNow, thetaNow, omegaNow = env.state()
    m_wet = math.exp(zNow)

    descendingVelYLimit = 5.0
    descendingAccYLimit = 0.3*abs(env.world.gravity.y) # The "velY" couldn't change too quickly in either direction 

    glide_tangent = math.tan(math.pi / 6) # Make this value SMALLER to allow WIDER cone
    z0 = np.zeros(N)
    mu1 = np.zeros(N)
    mu2 = np.zeros(N)
    try:
        for i in range(N):
            if m_wet - env.alpha * rho2 * i * gfold_dt <= 0:
                raise Exception('Solution not found for N = ' + str(N) + ' #1') 
            z0[i] = math.log(m_wet - env.alpha * rho2 * i * gfold_dt)
            mu1[i] = rho1 * math.exp(-z0[i])
            mu2[i] = rho2 * math.exp(-z0[i])

        r = opt.Variable((4, N), 'r')  # state vector (2 position,2 velocity)
        u = opt.Variable((2, N), 'u')  # u = thrust/mass
        z = opt.Variable(N, 'z')  # z = ln(mass)
        sigma = opt.Variable(N, 'sigma') # thrust slack parameter

        con = []  # CONSTRAINTS LIST
        con += [r[0:2, 0] == posNow]  # initial position
        con += [r[2:4, 0] == velNow]  # initial velocity

        rectified_y_target = rectified_y_target_factor*env.y_target
        rootLogger.warning(f'Proposed frames-of-flight is {N}, initial position is {env.lander.position}, target position is ({env.x_target}, {rectified_y_target}))')
        con += [ r[0:2, N-1] == (env.x_target, rectified_y_target) ]  # end position, set the target a little under the COG at landing spot
        con += [ r[2:4, N-1] == (0, 0) ]  # end velocity
        con += [ sigma[N-1] == 0 ] # end thrust

        con += [z[0] == zNow]  # initial mass

        for i in range(N - 1):
            if m_wet - env.alpha * rho1 * i * gfold_dt <= 0 or m_wet - env.alpha * rho2 * i * gfold_dt <= 0:
                raise Exception('Solution not found for N = ' + str(N) + ' #2') 

            """
            Don't go underground!
            """
            con += [ r[1, i] >= 0 ]

            """
            z_dot[i] = -alpha*sigma[i]
            where
            z_dot[i] = (z[i+1]-z[i])/dt
            """
            con += [ z[i + 1] == z[i] - gfold_dt * env.alpha * sigma[i] ]
            """
            r_dot_dot[i] = u[i] + g
            """
            con += [ r[2:4, i + 1] == r[2:4, i] + gfold_dt*(u[0:2, i] + np.array([env.world.gravity.x, env.world.gravity.y])) ]
            con += [ r[0:2, i + 1] == r[0:2, i] + gfold_dt*(r[2:4, i]) ]

            """
            |u[i]| <= sigma[i]
            """
            con += [opt.SOC(sigma[i], u[0:2, i])] # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t. 
            """
            mu1[i]*(1-(z[i]-z0[i])+(z[i]-z0[i])^2/2) <= sigma[i] <= mu2[i]*(1-(z[i]-z0[i]))
            """
            con += [mu1[i] * (1 - (z[i] - z0[i]) + 0.5*(z[i] - z0[i])**2) <= sigma[i],
                    sigma[i] <= mu2[i] * (1 - (z[i] - z0[i]))]
            """
            ln(m_wet-alpha*rho2*i*dt) <= z[i] <= ln(m_wet-alpha*rho1*i*dt)
            """
            con += [math.log(m_wet - env.alpha * rho2 * i * gfold_dt) <= z[i], z[i] <= math.log(m_wet - env.alpha * rho1 * i * gfold_dt)]
            """
            |(r[1, i]-env.y_target)/(r[0, i]-env.x_target)| >= certain tangent value

            such that the trajectory remains in a constrained cone through out all-time
            """
            #con += [ glide_tangent * opt.abs(r[0, i] - env.x_target) <= (r[1, i] - rectified_y_target) ] # the right side is always >= 0
            """
            [OPTIONAL, won't break SOCP compliance]
            "nozzle angle constraint" described at the end of the main paper as well as "Lossless Convexification of Nonconvex Control
    Bound and Pointing Constraints of the Soft Landing Optimal Control Problem" by Behçet Açıkmeşe and John M. Carson III, and Lars Blackmore.
            """
            con += [ sigma[i]*math.cos(NOZZLE_ABS_ANGLE_MAX_WRT_GROUND) <= u[1, i] ]
            """
            [OPTIONAL, won't break SOCP compliance]
            Shouldn't descend too fast at any point of time 
            """
            #con += [ r[1,i]-r[1,i+1] <= gfold_dt*descendingVelYLimit ]
            #con += [ opt.abs(r[3,i]-r[3,i+1]) <= gfold_dt*descendingAccYLimit ]
            con += [ r[1,i+1] - r[1,i] <= 0 ]

        mux.acquire()
        if foundNEvent.isSet() is True:
            mux.release()
            return

        mux.release()
        objective = opt.Maximize(z[N-1])
        problem = opt.Problem(objective, con)
        rootLogger.warning(f'problem types: is_dqcp={problem.is_dqcp()}, is_dcp={problem.is_dcp()}, is_qp={problem.is_qp()}')

        # [WARNING] The "problem" is SOCP and MUST be solvable for some N within [Nmin, Nmax], if not please check the convexity and SOCP compliance as well as the solver output by "verbose=True" carefully! 
        # For solver choices, see https://www.cvxpy.org/tutorial/advanced/index.html#solve-method-options
        problem.solve(solver=opt.ECOS, verbose=False) 

        if r.value is None:
            raise Exception('Solution not found for N = ' + str(N) + ' #3') 

        rootLogger.info(f'Found N = {N} is feasible')
        mux.acquire()
        resultHolder.append(N)
        resultHolder.append(r)
        resultHolder.append(u)
        resultHolder.append(z)
        resultHolder.append(sigma)
        foundNEvent.set()
        mux.release()

    except Exception as ex: 
        rootLogger.warning('Rejected N = ' + str(N) + ' due to exception: ' + str(ex))


if __name__ == "__main__":

    PHYSICS_FPS = 60
    physic_dt = 1.0 / PHYSICS_FPS
    physic_dt_millis = 1000.0 / PHYSICS_FPS
    
    initialPosList = [(52, 640.0), (520.0, 640.0)] 
    initialPos = initialPosList[1] 

    env = LunarLanderFuel(withNozzleImpluseToCOG=False, withFuelTank=True, withNozzleAngleConstraint=True, fps=PHYSICS_FPS, initialX=initialPos[0], initialY=initialPos[1])
    env.reset()
    # vid = video_recorder.VideoRecorder(env=env, path="./out/gfold.mp4")

    """
    PART #1

    The following notations are deliberately attempted to be aligned with  "Convex Programming Approach to Powered Descent Guidance for Mars Landing" by Behçet Açıkmeşe and Scott R. Ploen.

    The algorithm in this part is more commonly known as G-FOLD and only serves a "guidance path planning" purpose.  
    """

    posNow, velNow, zNow, thetaNow, omegaNow = env.state()
    m_wet = math.exp(zNow)

    total_reward = 0 # Not used yet

    DEGTORAD = math.pi / 180
    NOZZLE_ABS_ANGLE_MAX_WRT_GROUND = 45 * DEGTORAD # This is the angle measured w.r.t. ground frame

       
    """
    - rho1 & rho2 having unit "Newton", moreover
    - the "difference between rho1 & rho2" must be small enough such that for every "tf", all "t <= tf" suffice "m_wet - env.alpha * rho2 * t > 0" 
    """
    rho1 = env.rho1 
    rho2 = env.rho2
    rho_side_1 = env.rho_side_1
    rho_side_2 = env.rho_side_2

    tf_min = (env.initial_mass - env.remaining_fuel) * np.linalg.norm(velNow) / rho2
    tf_max = (env.remaining_fuel / (env.alpha * rho1))

    rootLogger.info(f'initial total mass: {env.initial_mass} kg, initial fuel mass: {env.remaining_fuel} kg, constant body mass: {env.initial_mass - env.remaining_fuel} kg, alpha: {env.alpha} kg/(second*Newton), rho1: {rho1} Newton, rho2: {rho2} Newton, rho_side_1: {rho_side_1} Newton, rho_side_2: {rho_side_2} Newton\nalpha*rho1: {env.alpha*rho1} kg/second, alpha*rho2: {env.alpha*rho2} kg/second, tf_min: {tf_min} seconds, tf_max: {tf_max} seconds')

    GFOLD_FPS = 15 # This value couldn't be set too low if using "gfoldFrameCnt = N-bisect.bisect_right(ys, posNow.y)" to find the proper "gfoldFrameCnt", otherwise it'll stick to the 0-th frame
    gfold_dt = 1.0 / GFOLD_FPS
    gfold_dt_millis = 1000.0 / GFOLD_FPS

    Nmin = int(tf_min*GFOLD_FPS)
    Nmax = int(tf_max*GFOLD_FPS)
    effectiveNmin = int((Nmin+Nmax)*0.20)
    effectiveNmax = Nmax+1

    rootLogger.info(f'Nmin: {Nmin}, Nmax: {Nmax}')
        
    resultHolder = []
    foundNEvent = Event()
    foundNEvent.clear()
    #pool = gevent.get_hub().threadpool
    for i in range(effectiveNmin, effectiveNmax):
        gevent.spawn(search_result, i, env, foundNEvent, resultHolder)

    timeoutSeconds = 120.0
    rootLogger.info(f'Starting to await resultHolder for {timeoutSeconds} seconds')
    foundNEvent.wait(timeout=timeoutSeconds)
    if foundNEvent.isSet() is False:
        rootLogger.error("No valid N is found!")
        sys.exit(0)
    
    N, r, u, z, sigma = resultHolder

    rootLogger.info("r[0:4, 0] is %s\nr[0:4, N-1] is %s\nmass[N-1] is %s\ntarget is (%s, %s)", np.array_str(r.value[0:4, 0]), np.array_str(r.value[0:4, N-1]), math.exp(z.value[N-1]), env.x_target, env.y_target)

    env.planned_trajectory = r.value[0:2, :]

    ys = r.value[1, :]
    ys = ys[::-1] # reverse to make ascending order

    """
    PART #2

    Try to follow the guidance path for landing. For reasons to choose the following dimensions of control please see [lunarlander_pid_only](./lunarlander_pid_only.py).
    """
    pss = Symbol('pss', real = True, positive = True) # percentage of steady state
    Kp = Symbol('Kp', real = True, positive = True)
    Kd = Symbol('Kd', real = True, positive = True)
    t = Symbol('t', real = True, positive = True)
        
    matcher, matcherEnvelope, tsConservative, tFirstPeak, firstPeakAmp, pOS = pd_designer.calcTimeDomainSymbolsForAcclerationModel(Kp, Kd, t, pss)

    # TODO: Load "tsConservative, pOS" from cache
    with open(f'{CWD}/gfold_exps.txt', 'wb') as f: 
        tsConservativeDump = dumps(lambdify((Kp, Kd), tsConservative)) 
        pOSDump = dumps(lambdify((Kp, Kd), pOS)) 
        f.write(tsConservativeDump)

    expected_ts = (N*gfold_dt*0.6) # seconds
    if expected_ts > 4.5: # clip expected settling time
        expected_ts = 4.5

    expected_pOS = 0.14
    percentageOfSteadyState0 = 0.05

    rootLogger.warning(f'expected_ts={expected_ts}, expected_pOS={expected_pOS}, percentageOfSteadyState0={percentageOfSteadyState0}')

    sol = pd_designer.calcKpAndKdForModel(Kp, Kd, t, pOS, pss, tsConservative, expected_ts, pOS0=expected_pOS, pss0=percentageOfSteadyState0)

    Kp0 = sol[0]
    Kd0 = sol[1]
    
    accyController = PID(Kp0, 0.005, Kd0)
    accxController = PID(Kp0, 0.005, Kd0) 
    betaZController = PID(131.12442477170202, 0.01, 21.175025483564053)

    startTime = int(time.time()*1000)
    now = startTime

    elapsedMsArr = []
    xPosArr = []
    yPosArr = []
    xDtArr = []
    yDtArr = []
    thetaArr = []
    massArr = []

    plannedXPosArr = []
    plannedYPosArr = []
    plannedXDtArr = []
    plannedYDtArr = []
    plannedMassArr = []

    gfoldFrameCnt = 0
    rectified_y_target = rectified_y_target_factor*env.y_target

    while (1):
        # Step through the simulation (1 step). Refer to Simulation Update in constants.py
        # [main_engine_power, side_engine_power, nozzle_angle]
        posNow, velNow, zNow, thetaNow, omegaNow = env.state()
        massNow = math.exp(zNow)
        nextNow = int(time.time()*1000)
        elapsedMs = nextNow - startTime
        elapsedMsFromLastFrame = nextNow - now

        #gfoldFrameCnt = int(elapsedMs/gfold_dt_millis)
        gfoldFrameCnt = N-bisect.bisect_right(ys, posNow.y)

        now = nextNow
        act = None
        planned_theta_now = 0*DEGTORAD # ALWAYS a constant WITHIN [-15 degs, +15 degs], note that even without the need to follow "planned_trajectory", the PID to track a constant "theta" by controlling "-phi" only works when "abs(theta)" is small (but why?).
        planned_r_now = r.value[0:2, N-1]
        planned_r_dot_now = r.value[2:4, N-1]
        planned_r_dot_dot_now = np.array([0, 0])
        planned_mass_now = math.exp(z.value[N-1])
        distYToTarget = abs(rectified_y_target-posNow.y)
        
        """
        To make the landing speed even smaller, it's possible to impose the commented condition.
        """
        #if gfoldFrameCnt < N and abs(rectified_y_target - posNow.y) > (100.0/env.SCALE):
        if gfoldFrameCnt < N:
            planned_r_now = r.value[0:2, gfoldFrameCnt]
            planned_r_dot_now = r.value[2:4, gfoldFrameCnt]     
            planned_r_dot_dot_now = np.array(planned_r_dot_now - r.value[2:4,gfoldFrameCnt-1] if (gfoldFrameCnt-1 >= 0) else [0, 0])/gfold_dt 
            planned_mass_now = math.exp(z.value[gfoldFrameCnt])

        elapsedMsArr.append(elapsedMs)
        xPosArr.append(posNow.x)
        yPosArr.append(posNow.y)
        xDtArr.append(velNow.x)
        yDtArr.append(velNow.y)
        thetaArr.append(thetaNow)
        massArr.append(massNow)

        plannedXPosArr.append(planned_r_now[0])
        plannedYPosArr.append(planned_r_now[1])
        plannedXDtArr.append(planned_r_dot_now[0])
        plannedYDtArr.append(planned_r_dot_now[1])
        plannedMassArr.append(planned_mass_now)

        x_error = (planned_r_now[0] - posNow.x)
        x_dt_error = (planned_r_dot_now[0] - velNow.x)

        y_error = (planned_r_now[1] - posNow.y)
        y_dt_error = (planned_r_dot_now[1] - velNow.y)

        theta_error = (planned_theta_now - thetaNow)
        theta_dt_error = (0 - omegaNow)

        accyComputed = accyController.compute_output(y_error, y_dt_error) + planned_r_dot_dot_now[1] 
        accxComputed = accxController.compute_output(x_error, x_dt_error) + planned_r_dot_dot_now[0]
        betaZComputed = betaZController.compute_output(theta_error, theta_dt_error)

        if accyComputed < env.world.gravity.y:
           accyComputed = (env.world.gravity.y*0.3) 

        (mainThrustComputed, sideThrustComputed, phiComputed, choiceOfFrame) = env.controlFrom3DOFEquationSolver(accxComputed, accyComputed, betaZComputed)

        # [WARNING] Clip the thrust magnitude
        mainThrustComputed = env.clipThrust(mainThrustComputed, rho1, rho2)

        # [WARNING] Clip the side thrust magnitude
        sideThrustComputed = env.clipThrust(sideThrustComputed, rho_side_1, rho_side_2, False)

        plantDataStr = env.plantDataStr()
        controllerDataStr = f'elapsed {elapsedMs} ms\ngfoldFrameCnt = {gfoldFrameCnt}\ny_error = {y_error}, y_dt_error = {y_dt_error}\nx_error = {x_error}, x_dt_error = {x_dt_error}\ntheta_error = {theta_error/DEGTORAD} degs, theta_dt_error = {theta_dt_error/DEGTORAD} degs/second\naccyComputed = {accyComputed}\naccxComputed = {accxComputed}\nbetaZComputed = {betaZComputed} degs/second^2'
        maneuverDataStr = f'mainThrustComputed = {mainThrustComputed}\nsideThrustComputed = {sideThrustComputed}\nphiComputed = {phiComputed/DEGTORAD} degs\n'

        if gfoldFrameCnt % 20 == 0:
            rootLogger.info(plantDataStr)
            rootLogger.info(controllerDataStr)
            rootLogger.warning(maneuverDataStr)

        if choiceOfFrame is not True:
            rootLogger.info(plantDataStr)
            rootLogger.info(controllerDataStr)
            rootLogger.warning(f'plantData = {plantDataStr}\ncontrollerData={controllerDataStr}\nno choice!')

        act = np.array([
            mainThrustComputed,
            sideThrustComputed,
            phiComputed
        ])
        s, reward, done, info = env.step(act)
        total_reward += reward  # Accumulate reward
        # -------------------------------------
        # Optional render
        env.render(mode='batch')
        # Optional render if using "vid"
        #vid.capture_frame()

        if done:
            rootLogger.info(f'When done, {plantDataStr}')
            break

    env.close()

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(elapsedMsArr, xPosArr, color='blue', label='measured')
    axs[0, 0].plot(elapsedMsArr, plannedXPosArr, linestyle='dashed', color='blue', label='planned')
    axs[0, 0].legend()
    axs[0, 0].set_title('x pos')

    axs[0, 1].plot(elapsedMsArr, xDtArr, color='purple', label='measured')
    axs[0, 1].plot(elapsedMsArr, plannedXDtArr, linestyle='dashed', color='purple', label='planned')
    axs[0, 1].legend()
    axs[0, 1].set_title('x velocity') 

    axs[0, 2].plot(elapsedMsArr, yPosArr, color='orange', label='measured')
    axs[0, 2].plot(elapsedMsArr, plannedYPosArr, linestyle='dashed', color='orange', label='planned')
    axs[0, 2].legend()
    axs[0, 2].set_title('y pos')

    axs[1, 0].plot(elapsedMsArr, yDtArr, color='crimson', label='measured')
    axs[1, 0].plot(elapsedMsArr, plannedYDtArr, linestyle='dashed', color='crimson', label='planned')
    axs[1, 0].legend()
    axs[1, 0].set_title('y velocity') # TODO: There's a significant delay in the following of "y velocity"

    axs[1, 1].plot(elapsedMsArr, massArr, color='olive', label='measured')
    axs[1, 1].plot(elapsedMsArr, plannedMassArr, linestyle='dashed', color='olive', label='planned')
    axs[1, 1].legend()
    axs[1, 1].set_title('mass')

    axs[1, 2].plot(elapsedMsArr, thetaArr, 'tab:green')
    axs[1, 2].set_title('theta')


    fig.tight_layout()
    plt.show(block=True)
