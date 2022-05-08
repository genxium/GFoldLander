import logging
import os
import sys

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.DEBUG)
rootLogger.propagate = False # disables console logging if not later explicitly added

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

CWD = os.path.dirname(os.path.abspath(__file__))
rootLogger.info('CWD=%s', CWD)
sys.path.append(CWD)

USE_SMALL_ANGLE_APPROX_FOR_PHI = False

import math
import numpy as np

from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, weldJointDef, contactListener)

from gym.envs.box2d.lunar_lander import LunarLanderContinuous
from gym.envs.classic_control import rendering

from scipy.optimize import root, fsolve

"""
Deliberately NOT using the magic number from `<proj-root>/constants.py`
"""

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY = [
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
]

LANDER_CONSTANT = 1  # Constant controlling the dimensions
LANDER_LENGTH = 34 / LANDER_CONSTANT
LANDER_RADIUS = 10 / LANDER_CONSTANT

FUEL_TANK_POLY = [
    (-14, +10), (-14, -5),
    (+14, -5), (+14, +10)
]

NOZZLE_POLY = [
    (-0.5*LANDER_RADIUS, 0               ), (+0.5*LANDER_RADIUS, 0               ),
    (-0.5*LANDER_RADIUS, +LANDER_LENGTH/8), (+0.5*LANDER_RADIUS, +LANDER_LENGTH/8)
]

"""LEGS"""
LEG_AWAY = 20
LEG_DOWN = 20
LEG_W, LEG_H = 3, 12
LEG_SPRING_TORQUE = 40

"""Forces, Costs, Torque, Friction"""

INITIAL_FUEL_MASS_PERCENTAGE = 0.7
MAIN_ENGINE_FUEL_COST = 5
SIDE_ENGINE_FUEL_COST = 1

SIDE_ENGINE_VERTICAL_OFFSET = 15  # y-distance away from the top of the lander
SIDE_ENGINE_HEIGHT = LANDER_LENGTH - SIDE_ENGINE_VERTICAL_OFFSET
SIDE_ENGINE_AWAY = 3.0

DEGTORAD = math.pi/180
NOZZLE_ANGLE_LIMIT = 5*DEGTORAD
NOZZLE_TORQUE = 500 / LANDER_CONSTANT

VIEWPORT_W = 700
VIEWPORT_H = 680

class ContactDetector(contactListener):
    """
    Creates a contact listener to check when the rocket touches down.
    """

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

"""
TODO

In reality, 
- fuel consumption should result in a position change of COG for the lander body
- some intermittent impulses can be used to simulate effect of wind.
"""
class LunarLanderFuel(LunarLanderContinuous):

    # printing toggles

    SCALE = 5.0   # affects how fast-paced the game is, forces should be adjusted as well
    withNozzleAngleConstraint = False
    withNozzleImpluseToCOG = False
    withFuelTank = False
    fps = 1
    rhoNormalizer = 1.0
    alpha = 1.0
    alphaPerFrame = 1.0
    alphaSidePerFrame = 1.0  
    unscaled_initial_x = 0
    unscaled_initial_y = 0
    landerRadius = LANDER_RADIUS
    landerLength = LANDER_LENGTH
    localOverallCenterOfMass = None

    def __init__(self, withNozzleImpluseToCOG=False, withNozzleAngleConstraint=False, withFuelTank=False, fps=30, initialX=0, initialY=0):
        self.unscaled_initial_x = initialX
        self.unscaled_initial_y = initialY
        self.withNozzleImpluseToCOG = withNozzleImpluseToCOG 
        self.withNozzleAngleConstraint = withNozzleAngleConstraint
        self.withFuelTank = withFuelTank
        self.fps = fps
        self.alpha = 0.0003*MAIN_ENGINE_FUEL_COST # the rate mass is decreased w.r.t. input "main engine power", should be independent upon "self.SCALE" -- because the lander is of constant density, if the area is larger the lander becomes heavier, thus given a certain acceleration magnitude the force calculated would also be larger  
        self.alphaSide = self.alpha # Deliberately made the same as "alpha" to feature the theory in "paper"
        self.alphaPerFrame = self.alpha/self.fps
        self.alphaSidePerFrame = self.alphaSide/self.fps

        super().__init__()

    def get_consumed_fuel(self):
        if self.lander is not None:
            return self.initial_mass - self.lander.mass

    def reset(self):
        self._destroy()
        self.world.contactListener = ContactDetector(self)
        self.game_over = False

        W = VIEWPORT_W / self.SCALE
        H = VIEWPORT_H / self.SCALE

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (128.0, 0.0, 0.0)
        self.moon.color2 = (128.0, 0.0, 0.0)

        # LANDER BODY
        initial_x = self.unscaled_initial_x/self.SCALE
        initial_y = self.unscaled_initial_y/self.SCALE
        rootLogger.info(f'unscaled_initial_x={self.unscaled_initial_x}, unscaled_initial_y={self.unscaled_initial_y}, initial_x={initial_x}, initial_y={initial_y}')

        defaultDensity = 70.0 # Making the total wet mass about 2675kg, like that of "XL-1 Vehicle Overview" according to https://explorers.larc.nasa.gov/2019APSMEX/MO/pdf_files/Masten%20Lunar%20Delivery%20Service%20Payload%20Users%20Guide%20Rev%201.0%202019.2.4.pdf
        scaledVertices = [(x/self.SCALE, y/self.SCALE) for x,y in LANDER_POLY]
        landerPolyShape = polygonShape(
            vertices=scaledVertices
        )
        xLander = [coord[0] for coord in scaledVertices]
        yLander = [coord[1] for coord in scaledVertices]
        landerArea = 0.5*np.abs(np.dot(xLander,np.roll(yLander,1))-np.dot(yLander,np.roll(xLander,1)))
        defaultTotMass = defaultDensity*landerArea

        self.initial_mass = defaultTotMass # To make the "initial_mass" the same regardless of "self.withFuelTank"
        self.initial_fuel = (INITIAL_FUEL_MASS_PERCENTAGE * self.initial_mass)
        self.remaining_fuel = self.initial_fuel

        self.landerDensity = defaultDensity

        if self.withFuelTank is True:
            self.landerDensity = (self.initial_mass-self.initial_fuel)/landerArea

        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=landerPolyShape,
                density=self.landerDensity,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)

        # LEGS
        legs_tilt_abs_lower_limit_degs = 20
        legs_tilt_abs_upper_limit_degs = 25
        self.legs = []
        for i in [-1, +1]:
            # "-1 == i" is the right leg
            leg = self.world.CreateDynamicBody(
                position=((VIEWPORT_W*0.5 - i * LEG_AWAY) / self.SCALE, initial_y),
                angle= -i*legs_tilt_abs_lower_limit_degs*DEGTORAD, # [WARNING] The initial value MUST BE WITHIN [lowerAngle, upperAngle]! Otherwise weird behaviour would occur! 
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / self.SCALE, LEG_H / self.SCALE)),
                    density=0.001, # Making mass very small
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / self.SCALE, LEG_DOWN / self.SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
                lowerAngle = (legs_tilt_abs_lower_limit_degs * DEGTORAD) if -1 == i else (-legs_tilt_abs_upper_limit_degs * DEGTORAD),
                upperAngle = (legs_tilt_abs_upper_limit_degs * DEGTORAD) if -1 == i else (-legs_tilt_abs_lower_limit_degs * DEGTORAD),
            )

            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        # FUEL TANK
        if self.withFuelTank is True:
            fueltankArea = (28*15)/(self.SCALE*self.SCALE)
            self.fueltank = self.world.CreateDynamicBody(
                position=(initial_x, initial_y),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in FUEL_TANK_POLY]),
                    density=(self.initial_fuel/fueltankArea),
                    friction=0.1,
                    categoryBits=0x0040,
                    maskBits=0x001,  # collide only with ground
                    restitution=0.0)  # 0.99 bouncy
            )
            self.fueltank.color1 = (128.0, 0.0, 0.0)
            self.fueltank.color2 = (128.0, 0.0, 0.0)
            weldjd = weldJointDef(
                bodyA=self.lander,
                bodyB=self.fueltank,
                localAnchorA=(0, 0),
                localAnchorB=(0, 0)
            )
            self.fueltank.joint = self.world.CreateJoint(weldjd)

        # NOZZLE
        self.nozzle = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / self.SCALE, y / self.SCALE) for x, y in NOZZLE_POLY]),
                density=0.001, # Making mass very small
                friction=0.1,
                categoryBits=0x0040,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.nozzle.color1 = (0, 128.0, 0)
        self.nozzle.color2 = (0, 128.0, 0)
        rjd = revoluteJointDef(
            bodyA=self.lander,
            bodyB=self.nozzle,
            localAnchorA=(0, 0),
            localAnchorB=(0, 0.35), # Increase the y-anchor a bit to see the green nozzle
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=NOZZLE_TORQUE,
            motorSpeed=0
        )
        # [WARNING] `NOZZLE_ANGLE_LIMIT` is currently imposed by the `_step` function 
        # The default behaviour of a revolute joint is to rotate without resistance.
        self.nozzle.joint = self.world.CreateJoint(rjd)

        self.planned_trajectory = []
        # ----------------------------------------------------------------------------------------
        self.drawlist = self.legs + ([self.nozzle, self.lander, self.fueltank] if self.withFuelTank is True else [self.nozzle, self.lander])

        self.x_target = VIEWPORT_W / self.SCALE / 2
        self.y_target = (self.helipad_y + LEG_DOWN / self.SCALE)

        self.rho1 = 0.3*(-self.initial_mass*self.world.gravity.y)
        self.rho2 = 1.0*(-self.initial_mass*self.world.gravity.y)

        self.rho_side_1 = 0.1*self.rho1
        self.rho_side_2 = 0.1*self.rho2

        return self.step(np.array([0, 0, 0]))[0]

    def step(self, action):
        reward = 0
        done = False
        # Returning a state of the lunarlander [r, r_dot, z] = [r, r_dot, ln(m)]
        state = None

        # Main Force Calculations
        if self.remaining_fuel <= 0:
            rootLogger.info("You're out of fuel!")
            done = True
        else:
            if self.withNozzleAngleConstraint is True:
                inputNozzleAngle = np.clip(float(action[2]), -NOZZLE_ANGLE_LIMIT, NOZZLE_ANGLE_LIMIT)
            else:
                inputNozzleAngle = action[2]

            self.nozzle.angle = self.lander.angle + inputNozzleAngle

            mainImpulse = self.__main_engines_force_computation(action, nozzle=self.nozzle, landerBody=self.lander)
            sideImpulse, engine_dir = self.__side_engines_force_computation(action)

            self._decrease_mass(mainImpulse, sideImpulse)

            self.world.Step(1.0 / self.fps, 6 * 30, 2 * 30)

            state = self.state()

            if self.legs[0].ground_contact is True and self.legs[1].ground_contact is True:
                rootLogger.info("You've touched ground!")
                done = True

            if not self.lander.awake:
                done = True
                reward = +100

        return state, reward, done, {}

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/self.SCALE, 0, VIEWPORT_H/self.SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))
            obj.color2 = (max(0.2, 0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0,0,0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/self.SCALE
            self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(1,1,1) )
            self.viewer.draw_polygon( [(x, flagy2), (x, flagy2-10/self.SCALE), (x+25/self.SCALE, flagy2-5/self.SCALE)], color=(0.8,0.8,0) )

        if self.planned_trajectory is not None:
            self.viewer.draw_polyline([(xx, yy) for xx, yy in zip(self.planned_trajectory[0, :], self.planned_trajectory[1, :])], linewidth=2, color=(1, 0, 0))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def __main_engines_force_computation(self, action, nozzle, landerBody):
        # ----------------------------------------------------------------------------
        # Nozzle Angle Adjustment

        # For readability
        sin = -math.sin(nozzle.angle) # Note that the "nozzle.angle" is referencing the absolute 0 and within [0, 2*PI]
        cos = math.cos(nozzle.angle)

        # Random dispersion for the particles
        dispersion = [self.np_random.uniform(-0.05, +0.05) for _ in range(2)]

        # Main engine
        m_power = 0
        bodyImpulse = (0, 0)
        try:
            if (action[0] > 0.0):
                # Limits
                m_power = action[0]
                particle_ttl = (np.clip(m_power/self.alpha, 0.0, 1.0) + 1.0) * 0.6

                # 4 is move a bit downwards, +-2 for randomness
                ox = sin * (4 / self.SCALE + 2 * dispersion[0]) - cos * dispersion[1]
                oy = -cos * (4 / self.SCALE + 2 * dispersion[0]) - sin * dispersion[1]

                Tx = m_power*sin
                Ty = m_power*cos

                impulse_pos = (nozzle.position[0] + ox, nozzle.position[1] + oy)

                m_power_particle_mass = 2*self.SCALE # When SCALE is smaller the coordinate system is larger, i.e. the initial position is deemed further from target position, thus requiring the particles to be easier to push outwards 
                p = self._create_particle(m_power_particle_mass, impulse_pos[0], impulse_pos[1], particle_ttl, radius=2)

                nozzleImpulse = (-Tx, -Ty)
                bodyImpulse = (Tx, Ty)
                point = impulse_pos
                wake = True

                # Force instead of impulse. This enables proper scaling and values in Newtons
                p.ApplyForce(nozzleImpulse, point, wake)

                if self.withNozzleImpluseToCOG is True:
                    landerBody.ApplyForce(
                        bodyImpulse,
                        landerBody.worldCenter,
                        wake
                    )
                else:
                    # [WARNING] In this case the lander is prone to rotational error!
                    nozzle.ApplyForce(bodyImpulse, nozzle.worldCenter, wake)
        except Exception as ex:
            rootLogger.error('Error in main engine power. m_power=%s, ex=%s', m_power, ex)

        return bodyImpulse

    def __side_engines_force_computation(self, action):
        # ----------------------------------------------------------------------------
        # Side engines
        sin = math.sin(self.lander.angle)  # for readability
        cos = math.cos(self.lander.angle)
        y_dir = 1 # Positioning for the side Thrusters

        # Orientation engines
        engine_dir = np.sign(action[1])
        s_power = np.abs(action[1])
        particle_ttl = (np.clip(s_power/self.alpha, 0.0, 1.0) + 1.0) * 0.6

        # Positioning
        sideEngineH = SIDE_ENGINE_HEIGHT / self.SCALE
        dx_part1 = - sin * sideEngineH  # Used as reference for dy
        dx_part2 = - cos * engine_dir * SIDE_ENGINE_AWAY / self.SCALE
        dx = dx_part1 + dx_part2
        dy = np.sqrt(np.square(sideEngineH) - np.square(dx_part1)) * y_dir - sin * engine_dir * SIDE_ENGINE_AWAY / self.SCALE

        # Impulse Position
        impulse_pos = (self.lander.position[0] + dx, self.lander.position[1] + dy)
        # Plotting purposes only
        self.impulsePos = impulse_pos

        thrust_dir_unnormalized = (engine_dir*(impulse_pos[1] - self.lander.position[1]), engine_dir*(impulse_pos[0] - self.lander.position[0]))
        thrust_dir_mag = np.linalg.norm(thrust_dir_unnormalized)
        thrust_dir_normalized = np.array(thrust_dir_unnormalized)/thrust_dir_mag if thrust_dir_mag > 0 else np.array((0.0, 0.0))

        bodyImpulse = (thrust_dir_normalized[0] * s_power, thrust_dir_normalized[1] * s_power)
        nozzleImpulse = (-bodyImpulse[0], -bodyImpulse[1])

        try:
            s_power_particle_mass = 5*self.SCALE  
            p = self._create_particle(s_power_particle_mass, impulse_pos[0], impulse_pos[1], particle_ttl, 1)
            p.ApplyForce(nozzleImpulse, impulse_pos, True)
            self.lander.ApplyForce(bodyImpulse, impulse_pos, True)
        except Exception as ex:
            rootLogger.error("Error due to Nan in calculating y during sqrt(l^2 - x^2). "
                          "x^2 > l^2 due to approximations on the order of approximately 1e-15. %s", ex)

        return bodyImpulse, engine_dir

    def _decrease_mass(self, mainImpulse, sideImpulse):
        consumed_fuel = self.alphaPerFrame*np.linalg.norm(mainImpulse) + self.alphaSidePerFrame*np.linalg.norm(sideImpulse)
        self.remaining_fuel -= consumed_fuel

        if self.remaining_fuel < 0:
            self.remaining_fuel = 0
            if self.withFuelTank is True:
                self.fueltank.mass = 0.00001 # [Warning] Setting to true 0 will automatically bounce back to 1.0, could be a bug in Box2D
            else:
                self.lander.mass = (1-INITIAL_FUEL_MASS_PERCENTAGE) * self.initial_mass
        else:
            if self.withFuelTank is True:
                self.fueltank.mass = self.fueltank.mass - consumed_fuel 
                fuel_tank_new_height = 15*(self.remaining_fuel)/self.initial_fuel; 
                new_fuel_tank_poly = [
                    (-14, -5+fuel_tank_new_height), (-14, -5),
                    (+14, -5), (+14, -5+fuel_tank_new_height)
                ]
                self.fueltank.fixtures[0].shape.vertices = [(x / self.SCALE, y / self.SCALE) for x, y in new_fuel_tank_poly]
            else:
                next_lander_mass = self.lander.mass - consumed_fuel
                self.lander.mass = next_lander_mass


    def _create_particle(self, mass, x, y, ttl, radius=3):
        """
        Used for both the Main Engine and Side Engines
        :param mass: Different mass to represent different forces
        :param x: x position
        :param y:  y position
        :param ttl:
        :param radius:
        :return:
        """
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=radius / self.SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
        )
        p.ttl = ttl  # ttl is decreased with every time step to determine if the particle should be destroyed
        self.particles.append(p)
        # Check if some particles need cleaning
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particles):
        while self.particles and (all_particles or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def overall_center_Of_mass(self):
        if self.withFuelTank is True:
            return (self.lander.worldCenter*self.lander.mass + self.fueltank.worldCenter*self.fueltank.mass)/(self.lander.mass + self.fueltank.mass)
        else:
            return (self.lander.worldCenter)  

    def effective_mass(self):
        if self.withFuelTank is True:
            return (self.lander.mass + self.fueltank.mass)
        else:
            return self.lander.mass

    def estimated_Iz(self):
        scaledLanderRadius = self.landerRadius/self.SCALE
        return -self.effective_mass()*scaledLanderRadius*scaledLanderRadius

    def state(self):
        return [
            self.lander.position,
            self.lander.linearVelocity,
            math.log( self.effective_mass() ),
            self.lander.angle if self.lander.angle <= math.pi else self.lander.angle - 2*math.pi,
            self.lander.angularVelocity
        ]

    def controlFrom3DOFEquationSolver(self, accx, accy, betaZ):
        posNow, velNow, zNow, thetaNow, omegaNow = self.state()
        massNow = self.effective_mass()
        IzNow = self.estimated_Iz()
        g = -self.world.gravity.y
        scaledLanderRadius = self.landerRadius/self.SCALE

        H = scaledLanderRadius+scaledLanderRadius
        h = np.linalg.norm(self.overall_center_Of_mass() - self.nozzle.position)

        def controlTo3DOF(vec):
            """
            # vec = [Fe, Fs, φ], note that "Fe >= 0" is always expected

            -Fe*sin(θ+φ) + Fs*cosθ = accx*m
            Fe*cos(θ+φ) + Fs*sinθ - g*m = accy*m
            Fe*sinφ*h + Fs*(H-h) = βz*Iz
            """

            if USE_SMALL_ANGLE_APPROX_FOR_PHI is False:
                return [
                    -((vec[0])*math.sin(thetaNow)+vec[0]*math.cos(thetaNow)*math.sin(vec[2])) + vec[1]*math.cos(thetaNow) - accx*massNow,
                    (vec[0])*math.cos(thetaNow)*math.cos(vec[2]) - (vec[0])*math.sin(thetaNow)*math.sin(vec[2]) + vec[1]*math.sin(thetaNow) - (g+accy)*massNow,
                    (vec[0])*math.sin(vec[2])*h + vec[1]*(H-h) - betaZ*IzNow
                ]
            else:
                return [
                    -((vec[0])*math.sin(thetaNow)+(vec[0])*math.cos(thetaNow)*vec[2]) + vec[1]*math.cos(thetaNow) - accx*massNow,
                    (vec[0])*math.cos(thetaNow) - (vec[0])*math.sin(thetaNow)*vec[2] + vec[1]*math.sin(thetaNow) - (g+accy)*massNow,
                    (vec[0])*vec[2]*h + vec[1]*(H-h) - betaZ*IzNow
                ]

        errRateTolerence = 0.05
        sol, infodict, ier, mesg = fsolve(controlTo3DOF, np.array([massNow*(g+accy)/math.cos(thetaNow), 0, 0]), full_output=True)
        if ier != 1:
            return (0, 0, 0, None)

        err = controlTo3DOF(sol)
        #rootLogger.info(f'sol = {sol}, err = {err}')

        if abs(err[0]) > abs(errRateTolerence*accx*massNow) or abs(err[1]) > abs(errRateTolerence*(g+accy)*massNow) or abs(err[2]) > abs(errRateTolerence*betaZ*IzNow):
            return (0, 0, 0, None)

        if sol[0] < 0:
            sol[0] = 0 # it must be ensured that "Fe >= 0"

        return (sol[0], sol[1], sol[2], True)

    def rectifyInputNozzleAngle(self, phi):
        # 1st convert "phi" to [0, 2*PI] 
        phi = phi%(2*math.pi)
        # 2nd convert "phi" to [-PI, PI] such that the "nozzle angle limiting in the plant" works correctly 
        if phi > math.pi:
            phi = phi - 2*math.pi
        return phi

    def clipThrust(self, thrustComputed, rho1, rho2, shouldPrint=False):
        # it might be "thrustComputed < 0" for the side thrust   
        if shouldPrint is True:
            rootLogger.info(f'clipping {thrustComputed}')

        initMagnitude = abs(thrustComputed)
        if 0.05*rho1 > initMagnitude:
            return 0 # the proposed magnitude is too small 

        if initMagnitude < rho1:
            return thrustComputed*(rho1/initMagnitude)

        if initMagnitude > rho2:
            return thrustComputed*(rho2/initMagnitude)

        return thrustComputed

    def plantDataStr(self):
        return f'massNow={self.effective_mass()}, velNow={self.lander.linearVelocity}, thetaNow={self.lander.angle/DEGTORAD} degs, omegaNow={self.lander.angularVelocity/DEGTORAD} degs/s'
