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

import math
import numpy

def rmse(measured, guidance):
    ret = 0
    n = len(measured)
    for i,v in enumerate(measured):
        diff = (v-guidance[i])
        if isinstance(diff, (list, tuple, numpy.ndarray)):
            for singleDimension in diff:
                ret += singleDimension*singleDimension
        else:
            ret += diff*diff

    return math.sqrt(ret/n) 

def timeDiff(valArr, timeArr):
    result = []
    for i,f in enumerate(valArr):
        if i <= 0 or i >= len(valArr)-1:
            continue
        result.append((valArr[i+1]-valArr[i-1])/(timeArr[i]+timeArr[i+1]))   
    return result
