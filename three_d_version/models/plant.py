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

INITIAL_FUEL_MASS_PERCENTAGE = 0.7
MAIN_ENGINE_FUEL_COST = 5
SIDE_ENGINE_FUEL_COST = 1

from panda3d.core import Vec3
from panda3d.core import Point3
from panda3d.core import TransformState
from panda3d.core import BitMask32
from panda3d.core import Quat
from panda3d.core import LMatrix3

from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletCylinderShape
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import XUp, YUp, ZUp
from panda3d.bullet import BulletHingeConstraint, BulletGenericConstraint

# ParticleEffect required modules
from direct.particles.Particles import Particles
from direct.particles.ParticleEffect import ParticleEffect
from panda3d.physics import BaseParticleEmitter, BaseParticleRenderer

import math, numpy

epsl = 1e-5
epsh = 1e-8
DEGTORAD = math.pi / 180

# sympy deps, mainly for using the symbolic jacobian
import sympy
from sympy import Symbol, pprint, MatrixSymbol
from sympy.utilities.iterables import flatten
from sympy.utilities.lambdify import lambdify

CWD = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv(CWD + '/../.env')

sympyUseCache = os.getenv('SYMPY_USE_CACHE')
rootLogger.info(f'sympyUseCache = {sympyUseCache}')

import cvxpy as opt

ex, ey, ez = sympy.Matrix([1, 0, 0]), sympy.Matrix([0, 1, 0]), sympy.Matrix([0, 0, 1])
def crossproductSym(lhs, rhs):
    return ex * (lhs[1] * rhs[2] - lhs[2] * rhs[1]) + ey * (lhs[2] * rhs[0] - lhs[0] * rhs[2]) + ez * (
                lhs[0] * rhs[1] - lhs[1] * rhs[0])

exCvx, eyCvx, ezCvx = numpy.array([1, 0, 0]), numpy.array([0, 1, 0]), numpy.array([0, 0, 1])
def crossproductCvx(lhs, rhs):
    return exCvx * (lhs[1] * rhs[2] - lhs[2] * rhs[1]) + eyCvx * (lhs[2] * rhs[0] - lhs[0] * rhs[2]) + ezCvx * (
                lhs[0] * rhs[1] - lhs[1] * rhs[0])


class LunarLanderFuel:
    def __init__(self, withNozzleImpluseToCOG=False, withNozzleAngleConstraint=False, withFuelTank=False):
        self.withNozzleImpluseToCOG = withNozzleImpluseToCOG
        self.withNozzleAngleConstraint = withNozzleAngleConstraint
        self.withFuelTank = withFuelTank
        self.alpha = 0.0003 * MAIN_ENGINE_FUEL_COST  # the rate mass is decreased w.r.t. input "main engine power", should be independent upon "self.SCALE" -- because the lander is of constant density, if the area is larger the lander becomes heavier, thus given a certain acceleration magnitude the force calculated would also be larger
        self.alphaSide = self.alpha  # Deliberately made the same as "alpha" to feature the theory in "paper"

        super().__init__()
    
    def printPhysicsProps(self, bulletNode, name):
        #rootLogger.warning(f'{name}: linearVel = {bulletNode.get_linear_velocity()}, angularVel = {bulletNode.get_angular_velocity()}, totalForce = {bulletNode.getTotalForce()}, totalTorque = {bulletNode.getTotalTorque()}')
        pass

    def reset(self):
        self.gfoldFrameCnt = 0
        self.targetTorqueAndCogForceInBodyFrameAtGfoldFrameCnt = None
        self.minimizerMaxcv = None
        self.engineFmagList = [0.0]*len(self.enginePosList)

        # reposition the necessary parent NodePaths (excluding those just attached to them)
        self.physicsContainerNP.setPosHpr(self.initialPos, Vec3(0., 0., 0.))
        self.physicsContainerNP.node().setLinearVelocity(Vec3(0., 0., 0.))
        self.physicsContainerNP.node().setAngularVelocity(Vec3(0., 0., 0.))
        self.physicsContainerNP.node().clearForces()
        self.printPhysicsProps(self.physicsContainerNP.node(), 'body')

        for i, simLegNP in enumerate(self.simLegNPList):
            legAnchorOffset = Vec3(0., 0., 0.)
            legOffsetWrtPhysicsContainer = self.simLegOrigPosList[i]
            legAzimuthalOffsetWrtPhysicsContainer = simLegNP.getHpr()
            self.simLegColliderList[i].setPosHpr(self.physicsContainerNP.getPos() + legOffsetWrtPhysicsContainer, Vec3(0., 0., 0.))
            self.simLegColliderList[i].node().setLinearVelocity(Vec3(0., 0., 0.))
            self.simLegColliderList[i].node().setAngularVelocity(Vec3(0., 0., 0.))
            self.simLegColliderList[i].node().clearForces()
            self.printPhysicsProps(self.simLegColliderList[i].node(), f'leg-{i}')

        self.fuelTankNP.setPosHpr(self.physicsContainerNP.getPos(), Vec3(0., 0., 0.))
        self.fuelTankNP.node().setLinearVelocity(Vec3(0., 0., 0.))
        self.fuelTankNP.node().setAngularVelocity(Vec3(0., 0., 0.))
        self.fuelTankNP.node().clearForces()
        if self.withFuelTank is True:
            self.initialFuel = (INITIAL_FUEL_MASS_PERCENTAGE * self.initialMass)
            self.remainingFuel = self.initialFuel
            self.dryMass = (self.initialMass - self.initialFuel)
        else:
            self.initialFuel = 0.0
            self.remainingFuel = 0.0
            self.dryMass = self.initialMass
        self._updateFuelTankShape()
        self.printPhysicsProps(self.fuelTankNP.node(), f'fuelTank')

        for i, fmag in enumerate(self.engineFmagList):
            self.particleFactories[i].setLifespanBase(0.0)

    def cleanup(self):
        # Clear collection variables that would retain memory
        self.simLegOrigPosList, self.simLegColliderList, self.simLegNPList, self.simUpperEngineNPList, self.simLowerEngineNPList, self.simOtherPartsNPList = None, None, None, None, None, None

        self.simBodyFaces, self.simBodyVertices = None, None
        self.simLegFacesList, self.simLegVerticesList = None, None # List of lists
        self.windIKLegNPList = None
        self.windIKBodyFaces, self.windIKBodyVertices = None, None
        self.windIKLegFacesList, self.windIKLegVerticesList = None, None # List of lists

        self.particleFactories = None

        # For "Bullet3 node variables", it's unclear whether or not they need manual deallocation, maybe they'd be deallocated once "world" is set to None
        self.simBodyShape = None
        self.fuelTankShape = None
        self.fuelTankHinge = None

        self.peffect.cleanup()
        self.peffect = None

        # "Panda3D NodePath variables" don't need manual deallocation, they'll be deallocated once detached from "worldNP"
    def setupNumericalsOnly(self, initialMass, pos, targetPos, gDuringFlight):
        # fill in physical properties, i.e. "bullet3" properties 
        self.initialPos = pos
        self.initialMass = initialMass  # [POSSIBLY BUG of PyBullet]If too heavy would break the constraint of hinge rotation axis
        self.dryMass = 0.0
        self.initialFuel = 0.0
        self.remainingFuel = 0.0
        self.initFuelTankHeight = 0.5
        if self.withFuelTank is True:
            self.initialFuel = (INITIAL_FUEL_MASS_PERCENTAGE * self.initialMass)
            self.remainingFuel = self.initialFuel
            self.dryMass = (self.initialMass - self.initialFuel)
        else:
            self.initialFuel = 0.0
            self.remainingFuel = 0.0
            self.dryMass = self.initialMass

        # together with "self.enginePosList", deliberately made the thrusts not passing along either "upperEngineCenter" or "lowerEngineCenter" to ensure torque in any direction is possible
        # it's also by design that the top thrustors have CONCURRENT extended ray pair 
        upperThrustSkew = math.pi / 5
        self.thrustDirsBody = [
            Vec3(math.cos(upperThrustSkew), math.sin(upperThrustSkew), 0.0),
            Vec3(math.cos(math.pi - upperThrustSkew), math.sin(math.pi - upperThrustSkew), 0.0),
            Vec3(math.cos(math.pi + upperThrustSkew), math.sin(math.pi + upperThrustSkew), 0.0),
            Vec3(math.cos(2*math.pi - upperThrustSkew), math.sin(2*math.pi - upperThrustSkew), 0.0),

            Vec3(0., 0., 1.),
            Vec3(0., 0., 1.),
            Vec3(0., 0., 1.),
            Vec3(0., 0., 1.),
        ]

        lowerEngineXYScale = 1.0
        self.enginePosList = numpy.concatenate((
            [simNp.getPos() for simNp in self.simUpperEngineNPList],
            [Point3(lowerEngineXYScale*simNp.getPos()[0], lowerEngineXYScale*simNp.getPos()[1], simNp.getPos()[2]) for simNp in self.simLowerEngineNPList]
        ))
        #rootLogger.info(f'self.enginePosList =\n{self.enginePosList}\nself.thrustDirsBody =\n{self.thrustDirsBody}')

        self.engineFmagList = [0.0]*len(self.enginePosList)

    def setup(self, world, worldNP, initialMass, pos, targetPos, collisionMask, gDuringFlight):
        self.loadMeshForSimulation()
        self.setupNumericalsOnly(initialMass, pos, targetPos, gDuringFlight)
        self.world = world
        self.worldNP = worldNP
        np = worldNP.attachNewNode(BulletRigidBodyNode('RocketBody'))
        np.node().setMass(self.dryMass)
        np.node().addShape(self.simBodyShape)
        np.node().set_linear_sleep_threshold(0)
        np.node().set_angular_sleep_threshold(0)
        np.node().set_linear_damping(0)
        np.node().set_angular_damping(0)
        np.setPos(pos)
        np.setCollideMask(collisionMask)
        world.attach(np.node())

        self.physicsContainerNP = np  # For applying force & torque
        self.rho1 = 0.3 * (self.effective_mass() * gDuringFlight)
        self.rho2 = 1.0 * (self.effective_mass() * gDuringFlight)
        # Attach all displaying rocket parts to "self.physicsContainerNP", kindly note that all these parts were designed w.r.t. the origin of "self.simBodyNP"
        self.simBodyNP.reparentTo(self.physicsContainerNP)
        for i, part in enumerate(self.simOtherPartsNPList):
           part.reparentTo(self.physicsContainerNP)

        self.invIDiagBody = np.node().getInvInertiaDiagLocal()  # It's a constant in body frame, no need to guard by "self.mux"
        self.invIMatrixBody = LMatrix3(
            self.invIDiagBody[0], 0.0, 0.0,
            0.0, self.invIDiagBody[1], 0.0,
            0.0, 0.0, self.invIDiagBody[2]
        )
        self.ImatrixBody = LMatrix3()
        self.ImatrixBody.invertFrom(self.invIMatrixBody)
        #rootLogger.info(f'self.ImatrixBody =\n{self.ImatrixBody}')

        self.physicsContainerNPRefQuat = self.physicsContainerNP.getQuat()
        self.physicsContainerNPRefPos = targetPos
        self.physicsContainerNPRefVel = Vec3(0., 0., 0.)
        self.accErrQuat = Quat(1.0, 0.0, 0.0, 0.0)
        self.accErrPos = Point3(0.0, 0.0, 0.0)

        if self.withFuelTank is True:
            self.fuelTankNP = worldNP.attachNewNode(BulletRigidBodyNode('FuelTank'))
            self.fuelTankNP.node().setMass(self.remainingFuel)
            self.fuelTankNP.setPos(self.physicsContainerNP.getPos())
            self.fuelTankNP.setCollideMask(BitMask32(0x01))
            self.fuelTankShape = BulletCylinderShape(0.1, self.initFuelTankHeight, ZUp)
            self.fuelTankNP.node().addShape(self.fuelTankShape)
            self.fuelTankHinge = BulletHingeConstraint(
                self.physicsContainerNP.node(),
                self.fuelTankNP.node(),
                TransformState.makePosHpr(Point3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0)),
                TransformState.makePosHpr(Point3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0)),
                True)
            self.fuelTankHinge.enableAngularMotor(False, 0.0, 0.0)
            self.fuelTankHinge.enableMotor(False)
            self.fuelTankHinge.setBreakingThreshold(float('inf')) # By default it's about 3.4e+38
            #self.fuelTankHinge.setAngularOnly(False) # By default it's "False" and it's mandatory or otherwise the "fuelTankHinge" wouldn't adhere to the body
            self.fuelTankHinge.setLimit(0.0, 0.0) # I just want it to be a fixed constraint
            self.fuelTankHinge.setDebugDrawSize(2.0)
            # Both the BulletRigidBodyNode and BulletConstraint should be attached to the BulletWorld
            world.attach(self.fuelTankNP.node())
            world.attach(self.fuelTankHinge)

        # Physics Legs
        for i, simLegNP in enumerate(self.simLegNPList):
            leg = worldNP.attachNewNode(BulletRigidBodyNode(f'Leg-{i}'))
            self.simLegColliderList.append(leg)

            leg.node().setMass(2.0)

            legOffsetWrtPhysicsContainer = self.simLegOrigPosList[i]
            legAzimuthalOffsetWrtPhysicsContainer = simLegNP.getHpr()

            legCollisionShape = BulletTriangleMeshShape(self.simLegMesh, dynamic=True)
            legCollisionShapeTransform = TransformState.makeHpr(legAzimuthalOffsetWrtPhysicsContainer) # No need to translate as we'll reparent for each leg; By the time of writing, I don't know why this "legCollisionShapeTransform" couldn't be applied to "leg.setHpr(...)", i.e. would result in runtime error, hence applied here on the "shape"
            leg.node().addShape(legCollisionShape, legCollisionShapeTransform)

            #rootLogger.info(f'Of leg-{i} w.r.t. physicsContainer, position offset is {legOffsetWrtPhysicsContainer}, azimuthal offset is {legAzimuthalOffsetWrtPhysicsContainer}')
            leg.setPos(self.physicsContainerNP.getPos() + legOffsetWrtPhysicsContainer)

            # Definition of hpr/ypr rotation can be viewed at https://confluence.qps.nl/qinsy/latest/en/how-to-qinsy-rotation-matrices-151128766.html
            leg.setCollideMask(BitMask32(0x01))  # doesn't collide with "physicsContainerNP"

            # Hinge 
            # It's deliberate that the "(pivotA, axisA, pivotB, axisB)" constructor is NOT used here because it requires a weird "finally coincide (resultedAxisA, resultedAxisB) pair", and even if that requirement were satisfied the direction of rotation was DIFFICULT TO CONTROL. See "README.md" for more information.  
            # The "frameA" and "frameB" are assumed to be coinciding w.r.t. origins(which in turn being w.r.t. AnchorPoint of each) & axes at the begining, AFTER the previous "leg.setPos(...)" result!  
            sharedXRotationForFrames = 90.0  # When "swinging", it's the "z-axes from both frames" that should align and be swung around
            frameA = TransformState.makePosHpr(legOffsetWrtPhysicsContainer, 
                                                Vec3(legAzimuthalOffsetWrtPhysicsContainer[0], sharedXRotationForFrames, 0.)) # [WARNING] This is MOVING/ROTATING THE "frame/axes" NOT the physicsContainer itself!

            legAnchorOffset = Vec3(0., 0., 0.) # Use "legAnchorOffset[2]" to move the anchor of hinge "up" on the leg
            frameB = TransformState.makePosHpr(legAnchorOffset, Vec3(legAzimuthalOffsetWrtPhysicsContainer[0], sharedXRotationForFrames, 0.0))  # [WARNING] This is MOVING/ROTATING THE "frame/axes" NOT the leg itself! [WARNING] Need non-zero positive mass of each leg to "doPhysics(...)"!

            # Equivalent method 1 to compose a hinge constraint
            hinge = BulletHingeConstraint(np.node(), leg.node(), frameA, frameB, use_frame_a=True)
            hinge.enableAngularMotor(False, 0.0, 0.0)
            hinge.enableMotor(False)
            hinge.setLimit(0, 10.0)
            hinge.setBreakingThreshold(float('inf')) # By default it's about 3.4e+38

            # Equivalent method 2 to compose a hinge constraint
            #hinge = BulletGenericConstraint(np.node(), leg.node(), frameA, frameB, use_frame_a=True)
            #hinge.setLinearLimit(0, 0, 0)
            #hinge.setLinearLimit(1, 0, 0)
            #hinge.setLinearLimit(2, 0, 0)
            #hinge.setAngularLimit(0, 0, 0)
            #hinge.setAngularLimit(1, 0, 0)
            #hinge.setAngularLimit(2, 0, 10.0)

            hinge.setDebugDrawSize(2.0)
            # Both the BulletRigidBodyNode and BulletConstraint should be attached to the BulletWorld
            world.attach(leg.node())
            world.attach(hinge)

            self.simLegNPList[i].reparentTo(leg)  # We reposition and reparent each "displaying leg" to the "rigid body leg" such that the "displaying leg" can be animated by the physics engine
            self.simLegNPList[i].setPosHpr(frameB.getPos(), legAzimuthalOffsetWrtPhysicsContainer) # compensate for frameB position offset  

        self.gfoldFrameCnt = 0
        self.targetTorqueAndCogForceInBodyFrameAtGfoldFrameCnt = None
        self.minimizerMaxcv = None

        # displaying engines
        self.particleFactories = []
        self.peffect = ParticleEffect() # Kindly note that "ParticleEffect extends Panda3D NodePath", thus is the one to be attached in panda3d node tree
        self.peffect.setPos(0.000, 0.000, 0.000)
        self.peffect.setHpr(0.000, 0.000, 0.000)
        self.peffect.setScale(1.000, 1.000, 1.000)
        steamTex = loader.loadTexture('models/steam.png')
        fireTex = loader.loadTexture('models/fire.png')
        for i, uengineNP in enumerate(self.simUpperEngineNPList):
            p0 = Particles(
                f'uengine-{i}')  # [WARNING] Name of the "Particles" instance must be different to PREVENT the "ParticleEffect" instance from deeming them as a single instance in its dictionary!
            # Particles parameters
            p0.setFactory("PointParticleFactory")
            p0.setRenderer("SpriteParticleRenderer")
            p0.setEmitter("PointEmitter")
            p0.setPoolSize(64)
            p0.setBirthRate(0.050)
            p0.setLitterSize(10)
            p0.setLitterSpread(0)
            p0.setSystemGrowsOlderFlag(0)
            # Factory parameters
            p0.factory.setLifespanBase(0.000)
            p0.factory.setLifespanSpread(0.0000)
            p0.factory.setMassBase(1.0000)
            p0.factory.setMassSpread(0.1000)
            p0.factory.setTerminalVelocityBase(20.0000)
            p0.factory.setTerminalVelocitySpread(5.0000)
            self.particleFactories.append(p0.factory)
            # Renderer parameters
            p0.renderer.setAlphaMode(BaseParticleRenderer.PRALPHAOUT)
            p0.renderer.setUserAlpha(1.0)
            # Sprite parameters
            p0.renderer.setTexture(steamTex)
            p0.renderer.setXScaleFlag(1)
            p0.renderer.setYScaleFlag(1)
            p0.renderer.setAnimAngleFlag(0)
            p0.renderer.setInitialXScale(0.0020)
            p0.renderer.setFinalXScale(0.0002)
            p0.renderer.setInitialYScale(0.0020)
            p0.renderer.setFinalYScale(0.0002)
            p0.renderer.setNonanimatedTheta(0.0000)
            p0.renderer.setAlphaBlendMethod(BaseParticleRenderer.PPNOBLEND)
            p0.renderer.setAlphaDisable(0)
            # Emitter parameters
            p0.emitter.setEmissionType(BaseParticleEmitter.ETEXPLICIT)
            p0.emitter.setExplicitLaunchVector(self.thrustDirsBody[i]*-20.0)
            rotQuat = uengineNP.getQuat()
            p0.emitter.setAmplitude(0.5)
            enginePos = uengineNP.getPos()
            p0.nodePath.setPos(enginePos)
            p0.nodePath.setQuat(rotQuat)
            uengineNP.reparentTo(self.physicsContainerNP)
            self.peffect.addParticles(p0)

        for i, lengineNP in enumerate(self.simLowerEngineNPList):
            p0 = Particles(f'lengine-{i}')
            # Particles parameters
            p0.setFactory("PointParticleFactory")
            p0.setRenderer("SpriteParticleRenderer")
            p0.setEmitter("PointEmitter")
            p0.setPoolSize(512)
            p0.setBirthRate(0.050)
            p0.setLitterSize(10)
            p0.setLitterSpread(0)
            p0.setSystemGrowsOlderFlag(0)
            # Factory parameters
            p0.factory.setLifespanBase(0.000)
            p0.factory.setLifespanSpread(0.0000)
            p0.factory.setMassBase(1.0000)
            p0.factory.setMassSpread(0.1000)
            p0.factory.setTerminalVelocityBase(20.0000)
            p0.factory.setTerminalVelocitySpread(5.0000)
            self.particleFactories.append(p0.factory)
            # Renderer parameters
            p0.renderer.setAlphaMode(BaseParticleRenderer.PRALPHAOUT)
            p0.renderer.setUserAlpha(1.0)
            # Sprite parameters
            p0.renderer.setTexture(fireTex)
            p0.renderer.setXScaleFlag(1)
            p0.renderer.setYScaleFlag(1)
            p0.renderer.setAnimAngleFlag(0)
            p0.renderer.setInitialXScale(0.0020)
            p0.renderer.setFinalXScale(0.0020)
            p0.renderer.setInitialYScale(0.0020)
            p0.renderer.setFinalYScale(0.0020)
            p0.renderer.setNonanimatedTheta(0.0000)
            p0.renderer.setAlphaBlendMethod(BaseParticleRenderer.PPNOBLEND)
            p0.renderer.setAlphaDisable(0)
            # Emitter parameters
            p0.emitter.setEmissionType(BaseParticleEmitter.ETEXPLICIT)
            p0.emitter.setExplicitLaunchVector(self.thrustDirsBody[i+4]*-20.0)
            p0.emitter.setAmplitude(0.5)
            enginePos = lengineNP.getPos()
            rotQuat = lengineNP.getQuat()
            p0.nodePath.setPos(enginePos)
            p0.nodePath.setQuat(rotQuat)
            lengineNP.reparentTo(self.physicsContainerNP)
            self.peffect.addParticles(p0)
        self.peffect.start(
            self.physicsContainerNP,
            render # [IMPORTANT] To allow particles to follow inertia
        )

    def calcThrustorIKForControllerEstimation(self):
        # thruster effect estimation
        fmagListSym = [Symbol(f'Ts{i}', real=True) for i in range(len(self.enginePosList))]
        torqueSym = sympy.Matrix([0., 0., 0.])  # Constant sym expression in body frame
        cogForceSym = sympy.Matrix([0., 0., 0.])
        for i, leverVec in enumerate(self.enginePosList):
            leverVecSym = sympy.Matrix([leverVec[0], leverVec[1], leverVec[2]])
            thrustDirSym = sympy.Matrix(
                [self.thrustDirsBody[i][0], self.thrustDirsBody[i][1], self.thrustDirsBody[i][2]])
            thrustSym = fmagListSym[i] * thrustDirSym
            cogForceSym += thrustSym
            torqueSym += crossproductSym(leverVecSym, thrustSym)

        torqueAndCogForceSym = sympy.Matrix(
            [torqueSym[0], torqueSym[1], torqueSym[2], 
             cogForceSym[0], cogForceSym[1], cogForceSym[2]])

        torqueAndCogForceJacSym = torqueAndCogForceSym.jacobian(fmagListSym)
        # pprint(self.torqueAndCogForceJacSym)
        # pprint(self.torqueAndCogForceJacSym.rref())

        A = torqueAndCogForceJacSym 
        torqueAndCogForceSymPseudoInverse = ((A.transpose()*A)**-1)*A.transpose() # Reference https://www.sciencedirect.com/topics/engineering/overdetermined-system  
        #pprint(torqueAndCogForceSymPseudoInverse)
        torqueAndCogForcePseudoInverse = numpy.array(torqueAndCogForceSymPseudoInverse.tolist()).astype(numpy.float64)
        #rootLogger.info(torqueAndCogForcePseudoInverse)

        torqueAndCogForceJacNumpy = numpy.array(A.tolist()).astype(numpy.float64)

        torqueJacSym = torqueSym.jacobian(fmagListSym)
        cogForceJacSym = cogForceSym.jacobian(fmagListSym)

        # If "torqueAndCogForceJacSym" is a constant matrix (i.e. "torqueAndCogForce" is a linear system w.r.t. "fmagList"), then to make every "targetTorqueAndCogForce" obtainable from Inverse Kinematics, 'self.torqueAndCogForceJacSym.rank()' should be exactly 6! 
        rootLogger.info(
            f'rank of the torqueAndCogForceJacSym is {torqueAndCogForceJacSym.rank()}; rank of the torqueJacSym is {torqueJacSym.rank()}')

        # turn sympy expressions into lambdas
        torqueAndCogForceFromFmagList = lambdify([fmagListSym], flatten(torqueAndCogForceSym));
        torqueAndCogForceFromFmagListJac = lambdify([fmagListSym], torqueAndCogForceJacSym)

        torqueFromFmagList = lambdify([fmagListSym], flatten(torqueSym))
        torqueFromFmagListJac = lambdify([fmagListSym], torqueJacSym)
        cogForceFromFmagList = lambdify([fmagListSym], flatten(cogForceSym));
        cogForceFromFmagListJac = lambdify([fmagListSym], cogForceJacSym)

        ################################################
        # IK by cvxpy expressions
        ################################################
        fmagListCvx = opt.Variable(len(self.engineFmagList), 'fmagList')
        torqueAndCogForceCvx = torqueAndCogForceJacNumpy@fmagListCvx
        targetTorqueAndCogForceInBodyParamsCvx = opt.Parameter(6)
        objectiveCvx = opt.Minimize( opt.norm( torqueAndCogForceCvx - targetTorqueAndCogForceInBodyParamsCvx ) ) 
        conCvx = []
        """
        [WARNING] The following version DOESN'T work, because the constraint "(torqueAndCogForceCvx - targetTorqueAndCogForceInBodyParamsCvx) == 0" is linear and overdetermined. See https://www.sciencedirect.com/topics/engineering/overdetermined-system for details.  

        #objectiveCvx = opt.Minimize(opt.norm(fmagListCvx)) 
        """
        conCvx += [ (torqueAndCogForceCvx == targetTorqueAndCogForceInBodyParamsCvx) ] # This constraint is OPTIONAL, it could be violated -- yet it does help the solver run faster and approach better result (maybe due to cvxpy not accepting an "initGuess")
        for fmagCvx in fmagListCvx:
            conCvx += [fmagCvx >= 0.0] 
        problemCvx = opt.Problem(objectiveCvx, conCvx)
        
        return torqueAndCogForceFromFmagList, torqueAndCogForceFromFmagListJac, torqueFromFmagList, torqueFromFmagListJac, cogForceFromFmagList, cogForceFromFmagListJac, fmagListCvx, targetTorqueAndCogForceInBodyParamsCvx, problemCvx, torqueAndCogForcePseudoInverse   
         
    def calcWindEffectIKForControllerEstimation(self): 
        ###############################
        #
        # The method used by bullet3 (https://github.com/bulletphysics/bullet3/blob/master/src/BulletSoftBody/btSoftBody.cpp) is NOT DIRECTLY APPLICABLE here, because we SHOULD avoid conditional branching when calculating IK expressions, i.e. for controllable fins we should search for solutions w.r.t. SMOOTH IK expressions.
        #
        # A possibly BETTER method here could be the Kutta-Joukowski theorem.
        ###############################
        totForce = sympy.Matrix([0., 0., 0.])
        totTorque = sympy.Matrix([0., 0., 0.])

        windVelSym = MatrixSymbol('wv', 3, 1)  # wind velocity sym expression in body frame
        airDensitySym = Symbol('ad', real=True) 
        faces = self.windIKBodyFaces 
        vertices = self.windIKBodyVertices
        relVel = (-windVelSym) # in body frame, the rocket velocity is always (0, 0, 0)

        # I haven't found elegant way to calculate "relSpd2" and "relVelN" in sympy Matrics, sigh...
        relSpd2 = relVel[0]*relVel[0] + relVel[1]*relVel[1] + relVel[2]*relVel[2]  
        relSpd = sympy.sqrt(relSpd2)
        relVelN = relVel/relSpd

        nowQuat = self.physicsContainerNP.getQuat()
        for i, face in enumerate(faces):
            fDrag = sympy.Matrix([0., 0., 0.])
            fLift = sympy.Matrix([0., 0., 0.])
            faceCog = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]])/3
            facePerp = nowQuat.xform((vertices[face[1]]-vertices[face[0]]).cross(vertices[face[2]]-vertices[face[0]]))
            faceNorm = facePerp.normalized()
            faceArea = facePerp.length()*0.5
            
            """
            # [WARNING] Deliberately trying to avoid/lessen the use of symbolic "relVelN" in IK expressions!
            nDotV = faceNorm.dot(relVelN)  
            """
            faceDotRelV = (facePerp[0]*relVel[0] + facePerp[1]*relVel[1] + facePerp[2]*relVel[2])*0.5 # which is a scalar "nDotV*faceArea*relSpd" 
            faceCrossRelV = crossproductSym(facePerp*0.5, relVel) # which equals "faceNorm.cross(relVelN)*faceArea*relSpd" 

            kLF = self.controllerEstimatedKLF(facePerp, relVel)
            kDG = self.controllerEstimatedKDG(facePerp, relVel)

            #fDrag = (relVelN*(-1.0)*nDotV)*(faceArea*relSpd2*self.airDensity*kDG*0.5) # Deliberately put scalars at the end of expression
            fDrag = -relVel*faceDotRelV*(self.airDensitySym*kDG*0.5) 
            #fLift = (faceNorm.cross(relVelN).cross(relVelN))*(0.5 * kLF * self.airDensity * relSpd * faceArea * math.sqrt(1.0 - nDotV * nDotV))
            fLift = crossproductSym(faceCrossRelV, relVelN)*(0.5 * kLF * self.airDensitySym)

            forceInBody = (fLift + fDrag)
            totForce += forceInBody   
            totTorque += crossproductSym(faceCog, forceInBody)

        estimatedWindForceSym = totForce.simplify() 
        #pprint(self.estimatedWindForceSym)
        estimatedWindTorqueSym = totTorque.simplify()
        #pprint(self.estimatedWindTorqueSym)
        return estimatedWindForceSym, estimatedWindTorqueSym
       
    def _loadMeshForControllerWindIKEstimation(self):
        wholeRocketNP = loader.loadModel('models/controller_estimated_shape.bam')
        rocketPartNPList = list(wholeRocketNP.children.getPaths())
        rocketPartNPList.sort(key=lambda x:x.getName()) # [WARNING] It's critical to sort before using "self.torqueAndCogForceFromFmagList" for IK in "runPid" 
        #rootLogger.info(f'For controller IK, rocketPartNPList is {rocketPartNPList}')
        self.windIKBodyNP, self.windIKLegNPList = None, []
        self.windIKBodyFaces, self.windIKBodyVertices = [], None
        self.windIKLegFacesList, self.windIKLegVerticesList = [], [] # List of lists
        for i, rocketPartNP in enumerate(rocketPartNPList):
            if rocketPartNP.name == 'Body':
                # GeomNode reading reference https://docs.panda3d.org/1.10/python/programming/physics/bullet/collision-shapes, and the "bullet-samples/02_Shape" 
                self.windIKBodyNP = rocketPartNP
                bodyGeom = self.windIKBodyNP.findAllMatches('**/+GeomNode').getPath(0).node().getGeom(0)
                bodyMesh = BulletTriangleMesh()
                bodyMesh.addGeom(bodyGeom)
                self.windIKBodyVertices = bodyMesh.vertices
                bodyTriangles = bodyMesh.triangles
                for j, bodyTriangle in enumerate(bodyTriangles):
                    # each "bodyTriangle" stores the 3 indexes of vertices instead of the actual coordinates, to save memory for duplicate vertices
                    i0 = bodyTriangle[0]
                    i1 = bodyTriangle[1]
                    i2 = bodyTriangle[2]
                    self.windIKBodyFaces.append([i0, i1, i2])
                self.windIKBodyShape = BulletTriangleMeshShape(bodyMesh,
                                                    dynamic=True)  # [WARNING] MUST use "dynamic=True" to obtain a non-zero inertia tensor in body frame
            elif rocketPartNP.name.startswith('Leg'):
                self.windIKLegNPList.append(rocketPartNP)
                legGeom = rocketPartNP.findAllMatches('**/+GeomNode').getPath(0).node().getGeom(0)
                legMesh = BulletTriangleMesh()
                legMesh.addGeom(legGeom)
                self.windIKLegVerticesList.append(legMesh.vertices)
                legTriangles = legMesh.triangles
                legFaces = []
                for j, legTriangle in enumerate(legTriangles):
                    # each "bodyTriangle" stores the 3 indexes of vertices instead of the actual coordinates, to save memory for duplicate vertices
                    i0 = legTriangle[0]
                    i1 = legTriangle[1]
                    i2 = legTriangle[2]
                    legFaces.append([i0, i1, i2])
                self.windIKLegFacesList.append(legFaces)

    def loadMeshForSimulation(self):
        wholeRocketNP = loader.loadModel('models/rocket.bam')
        rocketPartNPList = list(wholeRocketNP.children.getPaths())
        rocketPartNPList.sort(key=lambda x:x.getName()) # [WARNING] It's critical to sort before using "self.torqueAndCogForceFromFmagList" for IK in "runPid" 
        #rootLogger.info(f'For simulation, rocketPartNPList is {rocketPartNPList}, typed {type(rocketPartNPList)}')
        self.simBodyNP, self.simLegOrigPosList, self.simLegColliderList, self.simLegNPList, self.simUpperEngineNPList, self.simLowerEngineNPList, self.simOtherPartsNPList = None, [], [], [], [], [], []
        self.simBodyShape = None
        self.simBodyFaces, self.simBodyVertices = [], None
        self.simLegFacesList, self.simLegVerticesList = [], [] # List of lists
        for i, rocketPartNP in enumerate(rocketPartNPList):
            if rocketPartNP.name == 'Body':
                # GeomNode reading reference https://docs.panda3d.org/1.10/python/programming/physics/bullet/collision-shapes, and the "bullet-samples/02_Shape" 
                self.simBodyNP = rocketPartNP
                bodyGeom = self.simBodyNP.findAllMatches('**/+GeomNode').getPath(0).node().getGeom(0)
                bodyMesh = BulletTriangleMesh()
                bodyMesh.addGeom(bodyGeom)
                self.simBodyVertices = bodyMesh.vertices
                bodyTriangles = bodyMesh.triangles
                for j, bodyTriangle in enumerate(bodyTriangles):
                    # each "bodyTriangle" stores the 3 indexes of vertices instead of the actual coordinates, to save memory for duplicate vertices
                    i0 = bodyTriangle[0]
                    i1 = bodyTriangle[1]
                    i2 = bodyTriangle[2]
                    self.simBodyFaces.append([i0, i1, i2])
                self.simBodyShape = BulletTriangleMeshShape(bodyMesh,
                                                    dynamic=True)  # [WARNING] MUST use "dynamic=True" to obtain a non-zero inertia tensor in body frame
            elif rocketPartNP.name.startswith('UEngine'):
                self.simUpperEngineNPList.append(rocketPartNP)
            elif rocketPartNP.name.startswith('LEngine'):
                self.simLowerEngineNPList.append(rocketPartNP)
            elif rocketPartNP.name.startswith('Leg'):
                self.simLegNPList.append(rocketPartNP)
                self.simLegOrigPosList.append(rocketPartNP.getPos())
                legGeom = rocketPartNP.findAllMatches('**/+GeomNode').getPath(0).node().getGeom(0)
                legMesh = BulletTriangleMesh()
                legMesh.addGeom(legGeom)
                self.simLegVerticesList.append(legMesh.vertices)
                legTriangles = legMesh.triangles
                legFaces = []
                for j, legTriangle in enumerate(legTriangles):
                    # each "bodyTriangle" stores the 3 indexes of vertices instead of the actual coordinates, to save memory for duplicate vertices
                    i0 = legTriangle[0]
                    i1 = legTriangle[1]
                    i2 = legTriangle[2]
                    legFaces.append([i0, i1, i2])
                self.simLegFacesList.append(legFaces)
                self.simLegMesh = legMesh
            else:
                self.simOtherPartsNPList.append(rocketPartNP)
        
    def angleAxisOfBody(self):
        # Reference https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/transformation.py
        bodyACenterOfMassTransformBasis = self.physicsContainerNP.getMat().getUpper3()
        R = numpy.array(bodyACenterOfMassTransformBasis, dtype=numpy.float64, copy=False)
        R33 = R[:3, :3]
        # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
        w, W = numpy.linalg.eig(R33.T)
        i = numpy.where(abs(numpy.real(w) - 1.0) < epsl)[0]
        if not len(i):
            # By "Euler's rotation theorem", we must be able to find exact one eigenvector corresponding to eigenvalue=1
            raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
        direction = numpy.real(W[:, i[-1]]).squeeze()
        # rotation angle depending on direction
        cosa = (numpy.trace(R33) - 1.0) / 2.0
        if abs(direction[2]) > epsh:
            sina = (R[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
        elif abs(direction[1]) > epsh:
            sina = (R[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
        else:
            sina = (R[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
        angle = math.atan2(sina, cosa)

        # It's actually the same results as from "Quat.getAxis()" and "Quat.getAngle()" https://docs.panda3d.org/1.10/python/reference/panda3d.core.LQuaternionf
        return angle, direction

    def estimatedCOGPos(self):
        # in body frame
        return Point3(0.0, 0.0, 0.0)

    def effective_mass(self):
        return self.physicsContainerNP.node().getMass() + self.remainingFuel

    def effective_ImatrixBody(self):
        return self.ImatrixBody * (self.effective_mass() / self.dryMass)

    def kLF(self, angleOfAttackCosine, relVel):
        # Build a profile for the lift coefficient
        return 0.4

    def kDG(self, angleOfAttackCosine, relVel):
        # Build a profile for the drag coefficient
        return 0.6

    def controllerEstimatedKLF(self, facePerp, relVel):
        # Used in IK, could be a sympy expression, but now only using a constant
        return 0.4

    def controllerEstimatedKDG(self, facePerp, relVel):
        # Used in IK, could be a sympy expression, but now only using a constant
        return 0.6


    def applyThrustsDryrun(self, fmagList, gfoldFrameCnt):
        # [WARNING] NOT THREAD-SAFE, take care in the caller!
        if epsl >= self.remainingFuel:
            self.engineFmagList = [epsh] * len(self.enginePosList) # [WARNING] DON'T set mass to accurately "0.0", because it breaks the Bullet Physics calculation (by staying still in world frame regardless of the constraints)
        else:
            self.engineFmagList = fmagList

        self.gfoldFrameCnt = gfoldFrameCnt

    def applyThrusts(self, dt):
        nowQuat = self.physicsContainerNP.getQuat()
        for i, fmag in enumerate(self.engineFmagList):
            forceInBody = self.thrustDirsBody[i] * fmag
            forcePosInBody = Point3(self.enginePosList[i][0], self.enginePosList[i][1], self.enginePosList[i][2])
            forceInWorld = nowQuat.xform(forceInBody)
            forceOffsetWrtNodeInWorld = nowQuat.xform(forcePosInBody)
            self.physicsContainerNP.node().applyImpulse(forceInWorld*dt, forceOffsetWrtNodeInWorld)  # world frame
            self.particleFactories[i].setLifespanBase(fmag * 0.0001)  # TODO: Prepare an appropriate "N -> seconds" conversion factor

        if self.withFuelTank is True:
            self._decreaseMass(dt)

    def _decreaseMass(self, dt):
        consumedFuelPerSec = self.alpha * sum([self.engineFmagList[i] for i in range(4, 8)]) + self.alphaSide * sum([self.engineFmagList[i] for i in range(0, 4)])
        consumedFuel = consumedFuelPerSec*dt
        self.remainingFuel -= consumedFuel

        if self.remainingFuel < 0:
            self.remainingFuel = 0

        self._updateFuelTankShape()

    def _updateFuelTankShape(self):
        # Erwin, the author of "pybullet", suggests the following way of updating dimensions of a rigid body, see https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=2548
        if self.fuelTankShape is not None:
            self.fuelTankNP.node().removeShape(self.fuelTankShape)

        self.fuelTankNP.node().setMass(self.remainingFuel)
        fuelTankNewHeight = self.initFuelTankHeight * (self.remainingFuel / self.initialFuel)
        self.fuelTankShape = BulletCylinderShape(0.1, fuelTankNewHeight, ZUp)
        self.fuelTankNP.node().addShape(self.fuelTankShape)

    def clipThrust(self, fmagList, rho1, rho2, shouldPrint=False):
        if shouldPrint is True:
            rootLogger.info(f'clipping {fmagList}')

        newFmagList = [epsh]*len(fmagList)
        for i, fmag in enumerate(fmagList):
            if 0.05*rho1 > fmag:
                continue # the proposed magnitude is too small 
            elif fmag < rho1:
                newFmagList[i] = rho1
            elif fmag > rho2:
                newFmagList[i] = rho2
            else:
                newFmagList[i] = fmag

        return newFmagList

    def state(self):
        return [
            self.physicsContainerNP.getPos(),
            self.physicsContainerNP.node().get_linear_velocity(),
            math.log(self.effective_mass()),
            self.physicsContainerNP.getQuat(),
            self.physicsContainerNP.node().get_angular_velocity()
        ]

    def plantDataArr(self):
        nowPos, nowVel, nowLogm, nowQuat, nowOmega = self.state()
        osdText = []
        osdText.append('Mass  (kg)    :  ' + f'{self.effective_mass():+06.2f}')
        osdText.append('Pos   (m)     : (' + ','.join([f'{val:+06.1f}' for i, val in enumerate(nowPos)]) + ')')
        #osdText.append('Vel   (m/s)   : (' + ','.join([f'{val:+06.1f}' for i, val in enumerate(nowVel)]) + ')')
        #osdText.append('Omega (rad/s) : (' + ','.join([f'{val:+06.1f}' for i, val in enumerate(nowOmega)]) + ')')
        osdText.append('Quat          : (' + ','.join([f'{val:+05.3f}' for i, val in enumerate(nowQuat)]) + ')')
        osdText.append('UThrusts (N)  : (' + ','.join([f'{self.engineFmagList[i]:+08.1f}' for i in range(0, 4)]) + ')')
        osdText.append('LThrusts (N)  : (' + ','.join([f'{self.engineFmagList[i]:+08.1f}' for i in range(4, 8)]) + ')')
        return osdText

    def plantDataStr(self):
        return '\n'.join(self.plantDataArr()) 
