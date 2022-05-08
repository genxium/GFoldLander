# -*- coding: utf-8 -*-

"""
Please read and try out "pd_designer.py" before using this script! Many notations will be used WITHOUT ANY EXPLANATION HERE because it's assumed that you've read that prerequisite!
"""

import colorlog as logging
import os
import sys
import math

logFormatter = logging.ColoredFormatter("%(log_color)s %(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.DEBUG)
rootLogger.propagate = False # disables console logging if not later explicitly added

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)

from control import matlab
import sympy
from sympy import I, Symbol, Poly, init_printing, pprint, expand, collect, lambdify

def calcTimeDomainSymbolsForAcclerationModel(KpSym, t, pss):
    """
    This method assumes a simple P-control. 
    
    ClosedLoopTf = Kp/(s^2 + Kp), UnitStepInput = 1/s
    UnitStepOutput = Kp/[s*(s^2 + Kp)]
    
    which corresponds to

    a = +I*sqrt(Kp), b = -I*sqrt(Kp), with a (Kp) scale in the amplitude, yet we only care about "pOS" hence the scale is not important 
    """
    a = + I*sympy.sqrt(KpSym)
    b = - I*sympy.sqrt(KpSym)

    matcherPart1 = 1/KpSym
    #matcherPart2 = (a*sympy.exp(-b*t) - b*sympy.exp(-a*t))/((b-a)*KpSym)
    #pprint(matcherPart2)

    matcherPart2 = -(sympy.cos(sympy.sqrt(KpSym)*t))/(KpSym) # easily recognized from pprint result 

    # Calc `matcher`
    matcher = KpSym*matcherPart1 + KpSym*matcherPart2

    return matcher.simplify()   

"""
An "ideal controller" for trajectory tracking is with "0 deadtime, 0 overshoot, 0 risetime" which in the reality is non-existent, but we can tune towards this goal.  

The tuning method used here is proposed by "Sigurd Skogestad and Chriss Grimholt" in "The SIMC method for smooth PID controller
tuning", see https://folk.ntnu.no/skoge/publications/2012/skogestad-improved-simc-pid/old-submitted/simcpid.pdf for full details.
"""
def probeFOPDMParams(G, s, pss, *args, **kwargs):
    # We should've estimated a "first-order plus delay model (FOPDM)" from "P-controller closed loop test", see "section 2.2" of the paper.
    KpSym = Symbol('Kp') 
    t = Symbol('t')
    matcher = calcTimeDomainSymbolsForAcclerationModel(KpSym, t, pss)
    pprint(matcher) # If not converging we couldn't proceed with the steps below

    Kc0 = 0.5
    H_SIMC = Kc0
    formula = G*H_SIMC/(1+G*H_SIMC)

    numer_denoms = formula.as_numer_denom() 
    num = list(map(float, Poly(numer_denoms[0],s).all_coeffs()))
    den = list(map(float, Poly(numer_denoms[1],s).all_coeffs()))
    tf = matlab.tf(num, den)
    y, tSeq = matlab.step(tf, *args, **kwargs)

    Kc, theta, tau1 = None, None, None

    maximas = []
    minimas = []

    n = len(y)
    ys = 1

    iFirstOvershoot, iFirstUndershoot = None, None
    for i in range(0, len(y)):
        if i > 0 and i+1 < n: 
            prevY, nextY = y[i-1], y[i+1]
            if prevY < y[i] and nextY < y[i]:
                maximas.append((i, y[i]))  
                if iFirstOvershoot is None and y[i] > ys:
                    iFirstOvershoot = i
                    
            if prevY > y[i] and nextY > y[i]:
                minimas.append((i, y[i]))  
                if iFirstOvershoot is not None and iFirstUndershoot is None and y[i] < ys:
                    iFirstUndershoot = i
                    break

        
    tp = tSeq[iFirstOvershoot]
    Dyp = y[iFirstOvershoot]
    Dyu = y[iFirstUndershoot]
    yInfty = 0.45*(Dyp + Dyu)

    D = (Dyp - yInfty)/yInfty
    B = abs(1 - yInfty)/yInfty # Should be 0 here
    rootLogger.info(f'tp is {tp} seconds, Dyp is {Dyp}, Dyu is {Dyu}, yInfty is {yInfty}, D is {D}, B is {B}')

    A = 1.152*D**2 - 1.607*D + 1
    r = 2*A/B

    Kc = 1/(Kc0*B)
    deadtime = tp*(0.309 + 0.209*math.exp(-0.61*r))
    tau1 = r*deadtime

    return Kc, deadtime, tau1

def firstOrderApproxForDoubleIntegrating(Kc, deadtime, tau1, *args, **kwargs):
    Kd, Kp, Ki = 0, Kc, Kc/tau1  
    return Kd, Kp, Ki

"""
See "Double Integrating" in "Table 1" of the paper, for the 2nd order approximator we won't be using either "Kc" or "tau1" from the 1st order estimation, but keep the "deadtime" from there.

Here "tauC" is a tuning parameter that's independent of the 1st order estimation and needs MANUAL INPUT, usually but not always we start with "tauC = deadtime"
"""
def secondOrderApproxForDoubleIntegrating(deadtime, tauC, *args, **kwargs):
    KcSecondOrder = 0.25/(tauC + deadtime)**2 
    tauISecondOrder = 4*(tauC + deadtime) 
    tauDSecondOrder = 4*(tauC + deadtime) 

    # (Kd, Kp, Ki) result, see equation (1) or (30) of the paper, or equation (2) of https://folk.ntnu.no/skoge/publications/2016/grimholt-dycops-pid-double-integrating/chriss_grimholt_dycops.pdf
    
    Kd, Kp, Ki = KcSecondOrder*tauDSecondOrder, KcSecondOrder*(tauISecondOrder+tauDSecondOrder)/tauISecondOrder, KcSecondOrder/tauISecondOrder  
    return Kd, Kp, Ki
