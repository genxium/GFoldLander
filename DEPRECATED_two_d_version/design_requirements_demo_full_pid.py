# -*- coding: utf-8 -*-

import colorlog as logging
import os
import sys

logFormatter = logging.ColoredFormatter("%(log_color)s %(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]\n%(message)s")

rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.DEBUG)
rootLogger.propagate = False # disables console logging if not later explicitly added

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)

from control_and_ai import pid_designer

import sympy, cmath, math, scipy, numpy
from sympy import Symbol, Poly, init_printing, pprint, expand, collect, lambdify

import matplotlib.pyplot as plt
from control import matlab

init_printing(use_unicode = True) 

def stepResponse(formula, s, tauC, pssVal, *args, **kwargs):
    numer_denoms = formula.as_numer_denom() 
    num = list(map(float, Poly(numer_denoms[0],s).all_coeffs()))
    den = list(map(float, Poly(numer_denoms[1],s).all_coeffs()))
    tf = matlab.tf(num, den)
    y, tSeq = matlab.step(tf, *args, **kwargs)

    try:
        i10 = next(i for i in range(0,len(y)-1) if y[i]>=y[-1]*.10)
        Tr = round(tSeq[next(i for i in range(i10,len(y)-1) if y[i]>=y[-1]*.90)]-tSeq[i10], 2)
    except StopIteration:
        Tr = "unknown"
    try:
        Ts = round(tSeq[next(len(y)-i for i in range(2,len(y)-1) if abs(y[-i]/y[-1])>(1+pssVal))]-tSeq[0],2)
    except StopIteration:
        Ts = "unknown"
        
    if Tr == "unknown" or Ts == "unknown": 
        return 

    rootLogger.info(f'OS: {round((y.max()/y[-1]-1)*100,2)}%, tauC: {tauC}, Tr: {Tr}, Ts: {Ts}')

s = Symbol('s')

G = 1/(s**2) # This is a special case where the half-rule doesn't apply, therefore we use the ClosedLoop test to find the deadtime (a.k.a. "theta" in the paper)

percentageOfSteadyState0 = 0.05

"""
As "G = 1/s^2" being a very particular case here, none of the methods in "section 2.2" of the paper can evaluate its deadtime as a converging system.

- Given a unit step input, the "open-loop response" is divergent.
- Given a unit step input, the "P-controller closed-loop resp 1-cos(sqrt(Kp)*t)" is divergent. 
- The half-rule doesn't apply because there's no damping, i.e. the denominator doesn't comply with equation (11) of the paper.
"""
#KcFirstOrder, deadtime, tau1FirstOrder = pid_designer.probeFOPDMParams(G, s, percentageOfSteadyState0) 
#rootLogger.info(f'first order deadtime = {deadtime} seconds')

KcFirstOrder = 17.31464224598874 
deadtime = 0.5
tau1FirstOrder = KcFirstOrder/1.0023925447508668 

#Kd, Kp, Ki = pid_designer.firstOrderApproxForDoubleIntegrating(KcFirstOrder, deadtime, tau1FirstOrder)
#rootLogger.info(f'Kd = {Kd}, Kp = {Kp}, Ki = {Ki}')

for offset in range(1, 5000, 1):
    deadtimeSecondOrder = offset*0.05
    tauC = 0.5*(deadtimeSecondOrder)
    Kd, Kp, Ki = pid_designer.secondOrderApproxForDoubleIntegrating(deadtimeSecondOrder, tauC)
    #rootLogger.info(f'Kd = {Kd}, Kp = {Kp}, Ki = {Ki}')
    H = (Kd*s + Kp + Ki/s)
    ClosedLoopTf = (G*H)/(1 + G*H)
    stepResponse(
                    ClosedLoopTf, 
                    s,
                    tauC, 
                    percentageOfSteadyState0
                )
