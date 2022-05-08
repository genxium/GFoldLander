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

from models.plant import LunarLanderFuel
from panda3d.core import Point3, Vec3, Vec4
import direct.directbase.DirectStart
import math, numpy
from scipy.optimize import minimize, Bounds, LinearConstraint
import cvxpy as opt

landerInitPos = Point3(-40.0, -10.0, 5.0)

lander = LunarLanderFuel(withNozzleImpluseToCOG=False, withNozzleAngleConstraint=False, withFuelTank=True)

initMass = 2375.60
lander.loadMeshForSimulation()
lander.setupNumericalsOnly(initMass, landerInitPos, landerInitPos, 9.80665)

torqueAndCogForceFromFmagList, torqueAndCogForceFromFmagListJac, torqueFromFmagList, torqueFromFmagListJac, cogForceFromFmagList, cogForceFromFmagListJac, fmagListCvx, targetTorqueAndCogForceInBodyParamsCvx, ctrlproblem, torqueAndCogForcePseudoInverse = lander.calcThrustorIKForControllerEstimation()

targetTorqueAndCogForceInBody=numpy.array([-109.40679931640625, -266.3192138671875, -4.968011379241943, -16547.42578125, -10203.9072265625, 11788.0361328125])
prevTargetTorqueAndCogForceInBody=numpy.array([27.717700958251953, -71.8092269897461, -5.728072166442871, -12903.302734375, -7909.87255859375, 11650.24609375])

"""
###
#SLSQP test
###
"""
directPseudoInverseResult = torqueAndCogForcePseudoInverse@targetTorqueAndCogForceInBody 
scaleFactor = (1/(lander.initialMass*10))

def cmpslsqp2(fmagList):
    torqueAndCogForceResArr = torqueAndCogForceFromFmagList(fmagList);
    return (torqueAndCogForceResArr - targetTorqueAndCogForceInBody)

def cmpslsqp2Scaled(fmagList):
    torqueAndCogForceResArr = torqueAndCogForceFromFmagList(fmagList);
    return (torqueAndCogForceResArr - targetTorqueAndCogForceInBody)*scaleFactor

def cmpslsqp3(fmagList):
    # [WARNING] Meaningless as an equality constraint, i.e. the rocket won't move according to the solution.
    torqueAndCogForceResArr = torqueAndCogForceFromFmagList(fmagList);
    return numpy.array([torqueAndCogForceResArr[3], torqueAndCogForceResArr[4], torqueAndCogForceResArr[5]])

eq_cons = {
    'type': 'eq',
    'fun' : cmpslsqp2Scaled,  
}

minimizerObj1 = ( lambda fmagList: sum([(scaleFactor*fmag)**2 for fmag in fmagList]) )  
minimizerObj2 = ( lambda fmagList: numpy.linalg.norm(cmpslsqp2(fmagList)) )  
minimizerObj3 = ( lambda fmagList: numpy.linalg.norm(cmpslsqp3(fmagList)) )  
minimizerObj4 = ( lambda fmagList: numpy.linalg.norm([(scaleFactor*fmag) for fmag in fmagList]) )  
solMin = minimize(
    fun=minimizerObj2,  
    x0 = directPseudoInverseResult, 
    method='SLSQP',
    bounds=[(0.0, None)]*len(lander.enginePosList), 
    constraints=[eq_cons], 
    options={'maxiter': 1000, 'eps': 1e-3, 'disp': True},             
)

sol=solMin.x
minimizerMaxcv = numpy.linalg.norm(cmpslsqp2(sol))
rootLogger.warning(f'\nsol is {sol}\n\nminimizerMaxcv={minimizerMaxcv}')

"""
###
#CVX test
###
"""
#targetTorqueAndCogForceInBodyParamsCvx.value = prevTargetTorqueAndCogForceInBody 
targetTorqueAndCogForceInBodyParamsCvx.value = targetTorqueAndCogForceInBody  
def cmpcvx(fmagList):
    torqueAndCogForceResArr = torqueAndCogForceFromFmagList(fmagList);
    return (torqueAndCogForceResArr - targetTorqueAndCogForceInBodyParamsCvx.value)

epscv = 1e2 # constraint violation
epsminimizer = 1e-2
ctrlproblem.solve(solver=opt.SCS, eps=epsminimizer, scale=1e-05, warm_start=True, verbose=True) 
#ctrlproblem.solve(solver=opt.ECOS, verbose=True) 
#ctrlproblem.solve(solver=opt.ECOS, abstol=epsminimizer, reltol=epsminimizer, feastol=epscv, abstol_inacc=epsminimizer, reltol_inacc=epsminimizer, feastol_inacc=epscv, verbose=True) 
#ctrlproblem.solve(solver=opt.ECOS, feastol=epscv, reltol=epsminimizer, abstol=epsminimizer, verbose=True) 
if ctrlproblem.status not in ["infeasible", "unbounded"]:
    sol = [var.value for i,var in enumerate(fmagListCvx)]
    minimizerMaxcv = numpy.linalg.norm(cmpcvx(sol))
    rootLogger.info(f'ctrlproblem.status: {ctrlproblem.status}\n\tsol={sol}\n\tminimizerMaxcv={minimizerMaxcv}')
else:
    rootLogger.warning(f'ctrlproblem.status: {ctrlproblem.status}\n\tsolver stats: {ctrlproblem.solver_stats.extra_stats}\ntargetTorqueAndCogForceInBody={targetTorqueAndCogForceInBody.tolist()}\nprevTargetTorqueAndCogForceInBody={prevTargetTorqueAndCogForceInBody.tolist()}') 
