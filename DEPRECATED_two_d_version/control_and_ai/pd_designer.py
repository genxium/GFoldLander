# -*- coding: utf-8 -*-

import colorlog as logging
import os
import sys

logFormatter = logging.ColoredFormatter("%(log_color)s %(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.DEBUG)
rootLogger.propagate = False # disables console logging if not later explicitly added

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)

import sympy, cmath, math, scipy, numpy
from sympy import Symbol, Poly, init_printing, pprint, expand, collect, lambdify
from sympy.core.numbers import I

# Special console support is needed to display the unicodes correctly, e.g. "Courier + DeJaVu Sans(for Pseudographics: 2013-25C4;) + No monospace constraint + No auto height constraint" for ConEmu 
init_printing(use_unicode = True) 

def calcTimeDomainSymbolsForAcclerationModel(Kp, Kd, t, percentageOfSteadyState):
    """
    Here's the plant with OpenLoopGain 'G' to which we want to build a PD-controller to make the ClosedLoopGain suffice the given design requirements
    - percentageOfSteadyState
    - percentageOvershoot
    - expectedTsettling: seconds to first enter settling region

    s = Symbol('s')
    G = 1/(s^2)
    H = Kd*s + Kp
    ClosedLoopTf = G*H/(1+G*H)

    These are some design requirements NOT USED in this case.
    - Tpeak
    - Tr: seconds to rise == <only numerical solution is practically available>
    """

    """
    ClosedLoopTf = (Kd*s + Kp)/(s^2 + Kd*s + Kp), UnitStepInput = 1/s
    UnitStepOutput = (Kd*s + Kp)/[s*(s^2 + Kd*s + Kp)]

    which corresponds to

    d = Kp/Kd, a = [Kd+sqrt(Kd^2 - 4*Kp)]/2, b = [Kd-sqrt(Kd^2 - 4*Kp)]/2 where "Kd^2 - 4*Kp < 0", with a (Kd) scale in the amplitude.

    Therefore p = Kd/2, q = sqrt(4*Kp - Kd)/2.
    """
    d = Symbol('d', real = True)
    p = Symbol('p', real = True)
    q = Symbol('q', real = True)

    a = p + I*q
    b = p - I*q

    """
    # The following knowledge is used to avoid calculating the "InverseLaplaceTransform" by sympy which is quite SLOW.

    InverseLaplace((s + d)/(s*(s + a)*(s + b))) = [1/(ab)][d - b(d-a)*exp(-at)/(b-a) + a(d-b)*exp(-bt)/(b-a)]*Heaviside(t)
    """
    matcherPart1 = d/(a*b).simplify()

    ### Calc `mainWave` and `amplitudeEnvelope`
    foo = sympy.exp(-p*t)*(sympy.cos(-q*t) + I*sympy.sin(-q*t)) # sympy.exp(-a*t)
    bar = sympy.exp(-p*t)*(sympy.cos(+q*t) + I*sympy.sin(+q*t)) # sympy.exp(-b*t)
    matcherPart2 = (a*(d-b)*bar - b*(d-a)*foo)/((b-a)*a*b)
    origMatcherPart2 = matcherPart2
    matcherPart2 = collect(matcherPart2.expand(trig=True), [sympy.exp(-p*t)], evaluate=True).simplify()
    #pprint(matcherPart2, use_unicode = True)

    matcherPart2TrigsFac = -sympy.exp(-p*t)/((p**2 + q**2)*q)
    matcherPart2Trigs = matcherPart2/matcherPart2TrigsFac
    #pprint(matcherPart2Trigs, use_unicode = True)

    sinKey1 = sympy.sin(q*t)
    cosKey1 = sympy.cos(q*t)
    matcherPart2TrigsDict = collect(matcherPart2Trigs, [sinKey1, cosKey1], evaluate=False)
    v1 = matcherPart2TrigsDict[sinKey1]
    w1 = matcherPart2TrigsDict[cosKey1]
    phaseOffset1 = sympy.atan2(v1, w1)
    mainWave = sympy.cos(q*t - phaseOffset1) 
    
    """
    # The "conservative settling time" can be determined by intersecting "amplitudeEnvelope" with "y1 = -percentageOfSteadyState & y2 = +percentageOfSteadyState"
    """
    amplitudeEnvelope = sympy.sqrt(v1**2 + w1**2)*matcherPart2TrigsFac 
    """
    # The first "d(matcherPart2)/dt = 0" indicates the first peak
    """
    matcherPart2 = amplitudeEnvelope*mainWave

    # The denominator of "H" will be "s^2 + Kd*s + Kp", thus to make it real-pole-free, we must have "Kd^2 < 4*Kp"
    qpSubs = {d: Kp/Kd, p: Kd/2, q: sympy.sqrt(4*Kp - Kd**2)/2}

    # Calc `matcher`
    matcherUnsubbed = Kd*matcherPart1 + Kd*matcherPart2
    matcher = matcherUnsubbed.evalf(subs=qpSubs).simplify()

    # Calc `tsConservative`
    tsConservative = (sympy.log((percentageOfSteadyState**2 * q**2 * (p**2 + q**2)**2)/(Kd**2 * ((d*q)**2 + (d*q - p**2 - q**2)**2)))/(-2*p)).evalf(subs=qpSubs).simplify()

    # Calc `Overshoot`
    # matcherPart2Diff = matcherPart2.diff(t)
    # sinKey2 = sympy.sin(q*t - phaseOffset1)
    # cosKey2 = sympy.cos(q*t - phaseOffset1)
    # matcherPart2DiffTrigsDict = collect(matcherPart2Diff, [sinKey2, cosKey2], evaluate=False)
    # v2 = matcherPart2TrigsDict[sinKey2]
    # w2 = matcherPart2TrigsDict[cosKey2]
    ### Unfortunately the keys are too long to fit in "collect" implementation by the time of writing, therefore I use a hand calc version here.
    v2 = sympy.sqrt((d*q)**2 + (d*q - p**2 - q**2)**2)
    w2 = sympy.sqrt((d*q)**2 + (d*q - p**2 - q**2)**2)*p/q
    
    phaseOffset2 = sympy.atan2(v2, w2) # the derivative waveform is now cos(q*t - phaseOffset1 - phaseOffset2), where the zero crosses are "(PI/2 + n*PI) == (q*t - phaseOffset1 - phaseOffset2)" 
    
    tFirstPeak = ((0.5*math.pi + phaseOffset1 + phaseOffset2)/q).evalf(subs=qpSubs)
    firstPeakAmp = matcher.evalf(subs={t: tFirstPeak}).evalf(subs=qpSubs) 
    pOS = (firstPeakAmp/(Kd*matcherPart1) - 1).evalf(subs=qpSubs).simplify()
    matcherEnvelope = Kd*(matcherPart1 + amplitudeEnvelope).evalf(subs=qpSubs).simplify()

    return matcher, matcherEnvelope, tsConservative, tFirstPeak, firstPeakAmp, pOS   


def calcKpAndKdForModel(Kp, Kd, t, pOS, pss, tsConservative, ts0, pOS0, pss0=0.02, errRateTolerence=0.05):
    tsConservativeWithPss = tsConservative.evalf(subs={pss: pss0})

    funcA = lambdify((Kp, Kd), pOS - pOS0)
    funcB = lambdify((Kp, Kd), tsConservativeWithPss - ts0)

    def affiliateFunc(KpAndKd):
        return [
            funcA(KpAndKd[0], KpAndKd[1]),
            funcB(KpAndKd[0], KpAndKd[1])
        ]


    sol = [None, None]
    for KpCand in range(1, 1000, 1):
        trueKpCand = KpCand*0.1
        KdCand = math.sqrt(4*trueKpCand)*0.95
        solCand, infodict, ier, mesg = scipy.optimize.fsolve(affiliateFunc, [trueKpCand, KdCand], full_output=True)
        if 1 != ier:
            continue
        err = affiliateFunc(solCand)
        errRate = [abs(err[0]/pOS0), abs(err[1]/ts0)]
        rootLogger.info(f'solCand = {solCand}, ier = {ier}, errRate = {errRate}')
        if errRate[0] < errRateTolerence and errRate[1] < errRateTolerence:
            sol = solCand
            break

    return sol 
