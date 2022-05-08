from panda3d.core import loadPrcFileData 

loadPrcFileData('', 'win-size 1024 768')
loadPrcFileData('', 'bullet-enable-contact-events true') # Needed for receiving contact events 'bullet-contact-added' and 'bullet-contact-destroyed'

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

import math, numpy

import direct.directbase.DirectStart
from direct.showbase.DirectObject import DirectObject
from direct.showbase.InputStateGlobal import inputState
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletWorld
from panda3d.core import AmbientLight, TextNode
from panda3d.core import BitMask32, DirectionalLight
from panda3d.core import Point3, Vec3, Vec4

from models.plant import LunarLanderFuel
from models.terrain import Terrain
from models.wind import Wind
from scipy.optimize import fsolve, minimize, Bounds, LinearConstraint
import cvxpy as opt

# g = 0.0
g = 9.83
epsl = 1e-5
epsh = 1e-8
GFOLD_FRAME_INVALID = -1

class Game(DirectObject):

    def __init__(self):
        self.landerInitPos = Point3(-40.0, -10.0, 5.0)
        self.camOffset = Vec3(20.0, -20.0, 20.0)

        base.enableParticles()
        base.setBackgroundColor(0.1, 0.1, 0.8, 1)
        base.setFrameRateMeter(True)

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

        # Input
        self.accept('escape', self.doExit)
        self.accept('r', self.doReset)
        self.accept('f1', self.toggleWireframe)
        self.accept('f2', self.toggleTexture)
        self.accept('v', self.toggleDebug)
        self.accept('f5', self.doScreenshot)

        self.accept('n', self.decThrusts)
        self.accept('m', self.incThrusts)

        inputState.watchWithModifiers('yaw-', 'a')
        inputState.watchWithModifiers('yaw+', 'd')
        inputState.watchWithModifiers('pitch-', 's')
        inputState.watchWithModifiers('pitch+', 'w')
        inputState.watchWithModifiers('roll-', 'q')
        inputState.watchWithModifiers('roll+', 'e')

        # Task
        taskMgr.add(self.update, 'updateWorld')

        # Physics
        self.setup()

    # _____HANDLER_____

    def doExit(self):
        self.cleanup()
        sys.exit(1)

    def doReset(self):
        self.cleanup()
        self.setup()

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

    def decThrusts(self):
        self.uniformFmag -= self.uniformFmagNotch
        if self.uniformFmag < 0.0:
            self.uniformFmag = 0.0
        self.lander.applyThrustsDryrun([self.uniformFmag]*len(self.lander.enginePosList))

    def incThrusts(self):
        self.uniformFmag += self.uniformFmagNotch
        if self.uniformFmag >= self.uniformFmagMax:
            self.uniformFmag = self.uniformFmagMax
        self.lander.applyThrustsDryrun([self.uniformFmag]*len(self.lander.enginePosList))

    # ____TASK___

    def processInput(self, dt):
        rotation = Vec3(0, 0, 0)

        if inputState.isSet('yaw-'):   rotation.setZ(-5.0)
        if inputState.isSet('yaw+'):   rotation.setZ(+5.0)
        if inputState.isSet('pitch-'): rotation.setY(-5.0)
        if inputState.isSet('pitch+'): rotation.setY(+5.0)
        if inputState.isSet('roll-'):  rotation.setX(-5.0)
        if inputState.isSet('roll+'):  rotation.setX(+5.0)

        if rotation[0] != 0 or rotation[1] != 0 or rotation[2] != 0: 
            # Otherwise will severely impact the smoothness of "doPhysics" 
            self.lander.physicsContainerNP.setHpr(self.lander.physicsContainerNP.getHpr() + rotation)

    def update(self, task):
        # Make BulletWorld advance in fixed timestep regardless of "globalDt", see https://docs.panda3d.org/1.10/cpp/reference/panda3d.bullet.BulletWorld?highlight=bulletworld#_CPPv4N11BulletWorld10do_physicsE11PN_stdfloati11PN_stdfloat for reference
        globalDt = globalClock.getDt()
        maxSubSteps = 1
        simulationFixedDt = 1.0/60.0
        self.processInput(simulationFixedDt)
        # self.wind.applyAeroForceToFaces(self.lander)
        self.world.doPhysics(globalDt, maxSubSteps, simulationFixedDt)
        if self.isLanded is True:
            return task.cont

        self.inContactLegs = set()
        result = self.world.contactTest(self.terrain.terrainRigidBodyNP.node())
        if 0 < result.getNumContacts():
            for item in result.getContacts():
                if item.node0.name.startswith('Leg'):
                    self.inContactLegs.add(item.node0.name)
                if item.node1.name.startswith('Leg'):
                    self.inContactLegs.add(item.node1.name)

        if self.isLanded is False and 4 == len(self.inContactLegs):
            rootLogger.info(f'All 4 legs are in contact!')
            self.isLanded = True
            self.lander.applyThrustsDryrun([0.0]*len(self.lander.enginePosList), GFOLD_FRAME_INVALID)
            self.lander.applyThrusts(simulationFixedDt) # To stop thrusting anim
            return task.cont

        base.cam.setPos(self.lander.physicsContainerNP.getPos() + self.camOffset)
        base.cam.lookAt(self.lander.physicsContainerNP)

        self.lander.applyThrusts(simulationFixedDt)

        osdText = self.lander.plantDataArr()
        osdText.append('WindVel (m/s) : (' + ','.join([f'{self.wind.vel(self.lander)[i]:+04.1f}' for i in range(3)]) + ')')
        self.osdNode.setText('\n'.join(osdText))

        return task.cont

    def cleanup(self):
        self.world = None
        self.worldNP.removeNode()

    def setup(self):
        self.isLanded = False
        self.worldNP = render.attachNewNode('World')

        self.uniformFmag = 0.0

        # Onscreen TextNode, use white foreground color and transparent background color
        font = loader.loadFont('Cascadia.ttf') # Monowidth font   
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

        # World
        self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
        self.debugNP.node().showWireframe(True)
        self.debugNP.node().showConstraints(True)
        self.debugNP.node().showBoundingBoxes(False)
        self.debugNP.node().showNormals(True)
        #self.debugNP.show()

        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -g))
        self.world.setDebugNode(self.debugNP.node())

        self.terrain = Terrain(self.world, self.worldNP)
        """
        [WARNING] Deliberately NOT using the methods below, because "bullet-contact-added & bullet-contact-destroyed" DOESN'T seem to be a pair like "onContactBegin & onContactEnd", see the "why" prints in these callbacks, they'll be fired often which is quite CONFUSING
        """
        # self.accept('bullet-contact-added', self.onContactAdded)
        # self.accept('bullet-contact-destroyed', self.onContactDestroyed)
        # self.terrain.terrainRigidBodyNP.node().notifyCollisions(True)
        
        # Wind
        self.wind = Wind()    

        # Load markers around the initial position of the rocket to see whether "getElevation" is working as expected
        marker1DCount = 10
        markerStep = 0.5
        for i in range(marker1DCount):
            for j in range(marker1DCount):
                markerNP = loader.loadModel('models/box.egg')
                markerX = (self.landerInitPos[0])+i*markerStep - (marker1DCount/2)*markerStep
                markerY = (self.landerInitPos[1])+j*markerStep - (marker1DCount/2)*markerStep
                markerZ = self.terrain.calcRectifiedZ(markerX, markerY) + 0.5
                markerNP.setScale(0.3)
                markerNP.setPos(markerX, markerY, markerZ)
                markerNP.setCollideMask(BitMask32.allOff())
                markerNP.reparentTo(self.worldNP)

        # Lander (dynamic)
        self.lander = LunarLanderFuel(withNozzleImpluseToCOG=False, withNozzleAngleConstraint=False, withFuelTank=True)
        #initMass = 2375.60
        initMass = 23.7560
        self.lander.setup(self.world, self.worldNP, initMass, self.landerInitPos, self.landerInitPos, BitMask32(0x10), 9.80665)
        self.uniformFmagNotch = 100.0
        self.uniformFmagMax = self.lander.initialMass*abs(g)

        self.torqueAndCogForceFromFmagList, self.torqueAndCogForceFromFmagListJac, self.torqueFromFmagList, self.torqueFromFmagListJac, self.cogForceFromFmagList, self.cogForceFromFmagListJac, self.fmagListCvx, self.targetTorqueAndCogForceInBodyParamsCvx, self.ctrlproblem, self.torqueAndCogForcePseudoInverse = self.lander.calcThrustorIKForControllerEstimation()

        self.inContactLegs = set()
 
    # def onContactAdded(self, node1, node2):
    #     legName = None
    #     if node1.name == 'Heightfield' and node2.name.startswith('Leg'):
    #         legName = node2.name
    #
    #     if (node2.name == 'Heightfield' and node1.name.startswith('Leg')):
    #         legName = node1.name
    #
    #     if legName is None:
    #         return
    #
    #     if (legName in self.inContactLegs):
    #         rootLogger.info(f'{legName} already in inContactLegs, why adding?')
    #         return
    #
    #     self.inContactLegs.add(legName)
    #     if 4 == len(self.inContactLegs):
    #         rootLogger.info(f'All 4 legs are in contact!')
    #
    # def onContactDestroyed(self, node1, node2):
    #     legName = None
    #     if node1.name == 'Heightfield' and node2.name.startswith('Leg'):
    #         legName = node2.name
    #
    #     if (node2.name == 'Heightfield' and node1.name.startswith('Leg')):
    #         legName = node1.name
    #
    #     if legName is None:
    #         return
    #
    #     if not (legName in self.inContactLegs):
    #         rootLogger.info(f'{legName} NOT in inContactLegs, why destroying?')
    #         return
    #
    #     self.inContactLegs.remove(legName)

game = Game()
base.run()
