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

from control_and_ai import pd_designer

import sympy, cmath, math, scipy, numpy
from sympy import Symbol, Poly, init_printing, pprint, expand, collect, lambdify
from sympy.core.numbers import I

from control import matlab
import matplotlib.pyplot as plt

init_printing(use_unicode = True) 

def stepResponse(formula, s, pssVal, matcherCurve, matcherEnvelopeCurve, *args, **kwargs):
    numer_denoms = formula.as_numer_denom() 
    num = list(map(float, Poly(numer_denoms[0],s).all_coeffs()))
    den = list(map(float, Poly(numer_denoms[1],s).all_coeffs()))
    tf = matlab.tf(num, den)
    y, tSeq = matlab.step(tf, *args, **kwargs)
    ymatcher = [matcherCurve(ti) for ti in tSeq]
    yEnvelope = [matcherEnvelopeCurve(ti) for ti in tSeq]

    fig, onlyAxs = plt.subplots()
    onlyAxs.plot(tSeq, y, color='red', label='actual')
    onlyAxs.plot(tSeq, ymatcher, color='blue', linestyle='dashed', label='matcher')
    onlyAxs.plot(tSeq, yEnvelope, color='orange', linestyle='dotted', label='envelope')

    onlyAxs.set_title("Step Response")
    onlyAxs.grid()
    onlyAxs.set_xlabel("time (s)")
    onlyAxs.set_ylabel("y(tSeq)")
    onlyAxs.legend()
    info ="\nOS: %f%s"%(round((y.max()/y[-1]-1)*100,2),'%')
    rootLogger.info(f'y[-50] = {y[-50]}, y[-1]={y[-1]}, pssVal = {pssVal}')
    try:
        i10 = next(i for i in range(0,len(y)-1) if y[i]>=y[-1]*.10)
        Tr = round(tSeq[next(i for i in range(i10,len(y)-1) if y[i]>=y[-1]*.90)]-tSeq[i10], 2)
    except StopIteration:
        Tr = "unknown"
    try:
        Ts = round(tSeq[next(len(y)-i for i in range(2,len(y)-1) if abs(y[-i]-y[-1])/y[-1] > pssVal)]-tSeq[0],2)
    except StopIteration:
        Ts = "unknown"
        
    info += "\nTr: %s"%(Tr)
    info +="\nTs: %s"%(Ts)
    rootLogger.info(info)
    fig.tight_layout()
    plt.show(block=True)

pss = Symbol('pss', real = True, positive = True) # percentage of steady state
# Kp = Symbol('Kp', real = True, positive = True)
# Kd = Symbol('Kd', real = True, positive = True)
Kp = Symbol('Kp', positive = True)
Kd = Symbol('Kd', positive = True)
t = Symbol('t', real = True, positive = True)
matcher, matcherEnvelope, tsConservative, tFirstPeak, firstPeakAmp, pOS = pd_designer.calcTimeDomainSymbolsForAcclerationModel(Kp, Kd, t, pss)

expected_ts = 6.5 # seconds
expected_pOS = 0.138
percentageOfSteadyState0 = 0.05

sol = pd_designer.calcKpAndKdForModel(Kp, Kd, t, pOS, pss, tsConservative, expected_ts, pOS0=expected_pOS, pss0=percentageOfSteadyState0)

#Kp0 = sol[0]
#Kd0 = sol[1]
# I want to set the following values, but they result in "complex number pOS", why?
Kp0 = 3.0
Kd0 = 5.0
rootLogger.info(f'(Kp0, Kd0) = ({Kp0}, {Kd0})')

tFirstPeakPredicted = tFirstPeak.evalf(subs={Kp: Kp0, Kd: Kd0})
firstPeakAmpPredicted = firstPeakAmp.evalf(subs={Kp: Kp0, Kd: Kd0})
tsConservativePredicted = tsConservative.evalf(subs={Kp: Kp0, Kd: Kd0, pss: percentageOfSteadyState0})
envelopeAttsConservativePredicted = matcherEnvelope.evalf(subs={Kp: Kp0, Kd: Kd0, t: tsConservativePredicted})
pOSPredicted = pOS.evalf(subs={Kp: Kp0, Kd: Kd0})

rootLogger.info(f'\ntsConservativePredicted = {tsConservativePredicted}\nenvelopeAtTsPredicted = {envelopeAttsConservativePredicted}\ntFirstPeak = {tFirstPeakPredicted}\nfirstPeakAmp = {firstPeakAmpPredicted}\npOS = {100*pOSPredicted}%')

s = Symbol('s')
G = 1/(s**2)
H = (Kd0*s + Kp0)
ClosedLoopTf = G*H/(1+G*H)

pprint(matcher.evalf(subs={Kp: Kp0, Kd: Kd0}))
matcherCurve = lambdify(t, matcher.evalf(subs={Kp: Kp0, Kd: Kd0}))
#pprint(matcherEnvelope.evalf(subs={Kp: Kp0, Kd: Kd0}))
matcherEnvelopeCurve = lambdify(t, matcherEnvelope.evalf(subs={Kp: Kp0, Kd: Kd0}))

stepResponse(
                ClosedLoopTf, 
                s,
                percentageOfSteadyState0,
                matcherCurve,
                matcherEnvelopeCurve
            )
