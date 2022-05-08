import colorlog as logging
import os
import sys

from gym.wrappers.monitoring import video_recorder

logFormatter = logging.ColoredFormatter("%(log_color)s %(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]\n%(message)s")

rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.DEBUG)
rootLogger.propagate = False # disables console logging if not later explicitly added

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

CWD = os.path.dirname(os.path.abspath(__file__))

rootLogger.info('CWD=%s', CWD)

sys.path.append(CWD)

from environments.lunarlanderfuel import LunarLanderFuel
import numpy as np
import math, time
from control_and_ai.pid import PID
from matplotlib import pyplot as plt

if __name__ == "__main__":

    PHYSICS_FPS = 60
    physic_dt = 1.0 / PHYSICS_FPS
    physic_dt_millis = 1000.0 / PHYSICS_FPS

    env = LunarLanderFuel(withNozzleImpluseToCOG=False, withFuelTank=True, withNozzleAngleConstraint=True, fps=PHYSICS_FPS, initialX=520.0, initialY=640.0)
    env.reset()
    #vid = video_recorder.VideoRecorder(env=env, path="./out/vid.mp4")

    posNow, velNow, zNow, thetaNow, omegaNow = env.state()
    m_wet = math.exp(zNow)

    rho1 = env.rho1
    rho2 = env.rho2
    rho_side_1 = env.rho_side_1
    rho_side_2 = env.rho_side_2
    rootLogger.warning(f'rho1: {rho1} Newton, rho2: {rho2} Newton, rho_side_1: {rho_side_1} Newton, rho_side_2: {rho_side_2} Newton')

    total_reward = 0 # Not used yet

    DEGTORAD = math.pi / 180
    NOZZLE_ABS_ANGLE_MAX_WRT_GROUND = 45 * DEGTORAD # This is the angle measured w.r.t. ground frame

    """
    Use a 2+1 DOF staged-controller ((accx, accy), betaZ -> mainThrust, sideThrust, phi) given small theta & small phi approximation, where "(accx, accy) = force/mass" at any instant.

    If outputing the thrusts and gimbal directly, then as they're "cross coupled" with all channels of errors, e.g. mainThrust amplitude will affect all of (x_error, x_dt_error, y_error, y_dt_error, theta_error, theta_dt_error) in general cases, we have to guess each coupling factor, hence determining each (Kp, Ki, Kd) become very complicated and difficult to derive from "design requirements (usually but not always `PeakOvershoot, RiseTime, SettlingTime`, see mini tutorial for more info)". 
    """
    # Aliasing "os: PeakOvershoot, ts: Tsettling", note that the "ts" here is a "conservative estimation", thus the actual settling time would be less than it, and the velocity at settling should be near 0.
    # os=0.15, ts=3s, (Kp, Kd) = (3.64234513, 3.52917091)
    # os=0.15, ts=4s, (Kp, Kd) = (2.0488191370585684, 2.6468781854457357)
    # os=0.15, ts=5s, (Kp, Kd) = (1.3112442476988557, 2.11750254834227)
    accyController = PID(1.3112442476988557, 0.005, 2.11750254834227) 
    accxController = PID(1.3112442476988557, 0.005, 2.11750254834227)
    betaZController = PID(131.12442477170202, 0, 21.175025483564053) # os = 0.05, ts = 0.5s

    env.planned_trajectory = None

    planned_theta_now = 0*DEGTORAD # ALWAYS a constant WITHIN [-15 degs, +15 degs], note that even without the need to follow "planned_trajectory", the PID to track a constant "theta" by controlling "-phi" only works when "abs(theta)" is small (but why?).   
    startTime = int(time.time()*1000)
    now = startTime

    elapsedMsArr = []
    xPosArr = []
    cogXDiffArr = []
    yPosArr = []
    yVelArr = []
    cogYDiffArr = []
    thetaArr = []
    massArr = []

    frameCnt = 0
    while (1):
        # Step through the simulation (1 step). Refer to Simulation Update in constants.py
        # [main_engine_power, side_engine_power, nozzle_angle]
        posNow, velNow, zNow, thetaNow, omegaNow = env.state()
        centerOfMassNow = env.overall_center_Of_mass()
        massNow = env.effective_mass()
        nextNow = int(time.time()*1000)
        elapsedMs = nextNow - startTime
        elapsedMsFromLastFrame = nextNow - now
        now = nextNow
        frameCnt = frameCnt + 1
        elapsedMsArr.append(elapsedMs)
        xPosArr.append(posNow.x)
        cogXDiffArr.append((centerOfMassNow.x - posNow.x)/env.landerLength)
        yPosArr.append(posNow.y)
        yVelArr.append(velNow.y);
        cogYDiffArr.append((centerOfMassNow.y - posNow.y)/env.landerLength)
        thetaArr.append(thetaNow)
        massArr.append(massNow)

        x_error = (env.x_target - posNow.x)
        x_dt_ref = 0
        x_dt_error = (x_dt_ref - velNow.x)
    
        actual_y_target = env.y_target 
        y_error = (actual_y_target - posNow.y)
        y_dt_ref = 0
        if abs(y_error) > 2.0:
            y_dt_ref = -1.0
        elif abs(y_error) > 0.5:
            y_dt_ref = -0.5
        else:
            y_dt_ref = -0.1

        y_dt_error = (y_dt_ref - velNow.y) # We want the lander to land slowly in the y direction

        theta_error = (planned_theta_now - thetaNow) 
        theta_dt_error = (0 - omegaNow)

        accyComputed = accyController.compute_output(y_error, y_dt_error)
        accxComputed = accxController.compute_output(x_error, x_dt_error)
        betaZComputed = betaZController.compute_output(theta_error, theta_dt_error)

        if accyComputed < env.world.gravity.y:
           accyComputed = (env.world.gravity.y*0.3) # Starting from "free-fall" the "negative-y-acc" shouldn't be too large during the whole landing. 

        (mainThrustComputed, sideThrustComputed, phiComputed, choiceOfFrame) = env.controlFrom3DOFEquationSolver(accxComputed, accyComputed, betaZComputed)

        # [WARNING] Clip the thrust magnitude
        mainThrustComputed = env.clipThrust(mainThrustComputed, rho1, rho2)

        # [WARNING] Clip the side thrust magnitude
        sideThrustComputed = env.clipThrust(sideThrustComputed, rho_side_1, rho_side_2, False)

        platDataStr = env.plantDataStr()
        controllerDataStr = f'elapsed {elapsedMs} ms\nframeCnt = {frameCnt}\ny_target={posNow.y+y_error}, y_error = {y_error}, y_dt_error = {y_dt_error}\nx_target={posNow.x+x_error}, x_error = {x_error}, x_dt_error = {x_dt_error}\ntheta_error = {theta_error/DEGTORAD} degs, theta_dt_error = {theta_dt_error/DEGTORAD} degs/second\naccyComputed = {accyComputed}\naccxComputed = {accxComputed}\nbetaZComputed = {betaZComputed} degs/second^2'
        maneuverDataStr = f'mainThrustComputed = {mainThrustComputed}\nphiComputed = {phiComputed/DEGTORAD} degs\nsideThrustComputed = {sideThrustComputed}\n'

        if frameCnt % 20 == 0:
            rootLogger.info(platDataStr)
            rootLogger.info(controllerDataStr)
            rootLogger.warning(maneuverDataStr)

        if choiceOfFrame is not True:
            rootLogger.info(platDataStr)
            rootLogger.info(controllerDataStr)
            rootLogger.error(f'no choice!')
            break

        act = np.array([
            mainThrustComputed,
            sideThrustComputed,
            phiComputed
        ])
        s, reward, done, info = env.step(act)
        total_reward += reward  # Accumulate reward
        # -------------------------------------
        # Optional render if using "vid"
        env.render(mode='batch')
        # vid.capture_frame()

        if done:
            rootLogger.info(f'Total Reward:\t{total_reward}')
            break
    env.close()

    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(elapsedMsArr, xPosArr, color='blue')
    axs[0, 0].set_title('x pos')
    axs[0, 1].plot(elapsedMsArr, yVelArr, color='orange')
    axs[0, 1].set_title('y velocity')
    axs[0, 2].plot(elapsedMsArr, thetaArr, color='green')
    axs[0, 2].set_title('theta')

    axs[1, 0].plot(elapsedMsArr, massArr, color='cyan')
    axs[1, 0].set_title('mass')
    axs[1, 1].plot(massArr, cogXDiffArr, color='magenta')
    axs[1, 1].set_title('cogXDiff/L')
    axs[1, 1].invert_xaxis()
    axs[1, 2].plot(massArr, cogYDiffArr, color='brown')
    axs[1, 2].set_title('cogYDiff/L')
    axs[1, 2].invert_xaxis()
    fig.tight_layout()
    plt.show(block=True)
