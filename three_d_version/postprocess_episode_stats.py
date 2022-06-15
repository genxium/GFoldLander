# -*- coding: utf-8 -*-
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

import pickle
import matplotlib.pyplot as plt
from control_and_ai import error_stats
import numpy
import pathlib
from scipy.stats import pearsonr

# Get the list of all files and directories
dirpath = CWD + '/episode_stats'

with open(f'{dirpath}/pearsonr_all.txt', 'w+') as pearsonrFile: 
    nowEp = 0
    tfMeasuredTot, tfGuidanceTot, rmseMTot, rmseMDotTot, rmseRTot, rmseRDotTot, rmseQTot = 0., 0., 0., 0., 0., 0., 0.
    pearsonrTickAndQuatErrTot, pearsonrTickAndMdotErrTot, pearsonrMxCvAndQuatErrTot, pearsonrMxCvAndMdotErrTot, pearsonrMxCvAndTickRatioTot = 0., 0., 0., 0., 0.
    pearsonrTickAndQuatErrTotNep, pearsonrTickAndMdotErrTotNep, pearsonrMxCvAndQuatErrTotNep, pearsonrMxCvAndMdotErrTotNep, pearsonrMxCvAndTickRatioTotNep = 0, 0, 0, 0, 0
    for filename in os.listdir(dirpath): 
        ext = pathlib.Path(filename).suffix
        if '.stats' != ext:
            continue
        filepath = dirpath + f'/{filename}' 
        statsFile = open(filepath, 'rb')
        data = pickle.load(statsFile)
        statsFile.close()

        tfMeasured = data[0]
        tfGuidance = data[1]
        elapsedMsArr = data[2]
        posArr = data[3] 
        plannedPosArr = data[4] 
        velArr = data[5] 
        plannedVelArr = data[6] 
        massArr = data[7] 
        plannedMassArr = data[8] 
        quatArr = data[9] 
        globalDtArr = data[10]
        maxcvArr = data[11]
        phyDivPidStArr = data[12]
        phyDivPidEdArr = data[13]
        #rootLogger.info(f'tfMeasured = {tfMeasured}')

        # Remove the None elements
        posArr = [f for i,f in enumerate(posArr) if elapsedMsArr[i] is not None]
        plannedPosArr = [f for i,f in enumerate(plannedPosArr) if elapsedMsArr[i] is not None]
        velArr = [f for i,f in enumerate(velArr) if elapsedMsArr[i] is not None]
        plannedVelArr = [f for i,f in enumerate(plannedVelArr) if elapsedMsArr[i] is not None]
        massArr = [f for i,f in enumerate(massArr) if elapsedMsArr[i] is not None]
        plannedMassArr = [f for i,f in enumerate(plannedMassArr) if elapsedMsArr[i] is not None]

        quatArr = [f for i,f in enumerate(quatArr) if elapsedMsArr[i] is not None]
        globalDtArr = [f for i,f in enumerate(globalDtArr) if elapsedMsArr[i] is not None]
        maxcvArr = [f if f else 0. for i,f in enumerate(maxcvArr) if elapsedMsArr[i] is not None]

        phyDivPidStArr = [f if f else 0. for i,f in enumerate(phyDivPidStArr) if elapsedMsArr[i] is not None]
        phyDivPidEdArr = [f if f else 0. for i,f in enumerate(phyDivPidEdArr) if elapsedMsArr[i] is not None]

        elapsedMsArr = [f for f in elapsedMsArr if f is not None] 

        massDtArr = error_stats.timeDiff(massArr, globalDtArr)
        plannedMassDtArr = error_stats.timeDiff(plannedMassArr, globalDtArr)
        phyDivPidEdDtArr = error_stats.timeDiff(phyDivPidEdArr, globalDtArr)

        subplotIdx = 1
        xdata = [f[0] for f in posArr] 
        ydata = [f[1] for f in posArr]
        zdata = [f[2] for f in posArr]

        plannedXdata = [f[0] for f in plannedPosArr] 
        plannedYdata = [f[1] for f in plannedPosArr]
        plannedZdata = [f[2] for f in plannedPosArr]

        dxdata = [f[0] for f in velArr] 
        dydata = [f[1] for f in velArr]
        dzdata = [f[2] for f in velArr]

        plannedDxdata = [f[0] for f in plannedVelArr] 
        plannedDydata = [f[1] for f in plannedVelArr]
        plannedDzdata = [f[2] for f in plannedVelArr]

        # We might show a mix of 2D and 3D subplots, see https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html for more information.
        fig = plt.figure(figsize=(12, 8))
        figName = f'episode_plot_{nowEp+1:02d}'
        fig.canvas.manager.set_window_title(figName)
        nRows, nCols = 3, 3

        ## 3D quiver plotting
        #attitudeArr = [f.getUp() for i,f in enumerate(quatArr)]
        #axdata = [f[0] for i,f in enumerate(attitudeArr)] # attitude x 
        #aydata = [f[1] for i,f in enumerate(attitudeArr)] # attitude y
        #azdata = [f[2] for i,f in enumerate(attitudeArr)] # attitude z
        #ax = fig.add_subplot(nRows, nCols, subplotIdx, projection='3d')
        #ax.quiver(xdata, ydata, zdata, axdata, aydata, azdata, length=10.0)
        #subplotIdx += 1

        ## Position trajectories
        ax = fig.add_subplot(nRows, nCols, subplotIdx)
        ax.plot(elapsedMsArr, xdata, color='red', label='x-measured')
        ax.plot(elapsedMsArr, plannedXdata, color='red', linestyle='dashed', label='x-guidance')
        ax.plot(elapsedMsArr, ydata, color='green', label='y-measured')
        ax.plot(elapsedMsArr, plannedYdata, color='green', linestyle='dashed', label='y-guidance')
        ax.plot(elapsedMsArr, zdata, color='blue', label='z-measured')
        ax.plot(elapsedMsArr, plannedZdata, color='blue', linestyle='dashed', label='z-guidance')
        ax.legend()
        ax.set_title('Position history') 
        subplotIdx += 1

        ## Velocity trajectories
        ax = fig.add_subplot(nRows, nCols, subplotIdx)
        ax.plot(elapsedMsArr, dxdata, color='red', label='vx-measured')
        ax.plot(elapsedMsArr, plannedDxdata, color='red', linestyle='dashed', label='vx-guidance')
        ax.plot(elapsedMsArr, dydata, color='green', label='vy-measured')
        ax.plot(elapsedMsArr, plannedDydata, color='green', linestyle='dashed', label='vy-guidance')
        ax.plot(elapsedMsArr, dzdata, color='blue', label='vz-measured')
        ax.plot(elapsedMsArr, plannedDzdata, color='blue', linestyle='dashed', label='vz-guidance')
        ax.legend()
        ax.set_title('Velocity history') 
        subplotIdx += 1

        ## Mass trajectories
        ax = fig.add_subplot(nRows, nCols, subplotIdx)
        ax.plot(elapsedMsArr, massArr, color='red', label='m-measured')
        ax.plot(elapsedMsArr, plannedMassArr, color='red', linestyle='dashed', label='m-guidance')
        ax.legend()
        ax.set_title('Mass history') 
        subplotIdx += 1

        ## Mass Dt trajectories
        ax = fig.add_subplot(nRows, nCols, subplotIdx)
        ax.plot(elapsedMsArr[1:-1], massDtArr, color='red', label='m-dt-measured')
        ax.plot(elapsedMsArr[1:-1], plannedMassDtArr, color='red', linestyle='dashed', label='m-dt-guidance')
        ax.legend()
        ax.set_title('Mass Dt history') 
        subplotIdx += 1

        ## Quat deviation history
        quatDeviationArr = [((f.getR()-1.0)**2 + f.getI()**2 + f.getJ()**2 + f.getK()**2) for i,f in enumerate(quatArr)]
        ax = fig.add_subplot(nRows, nCols, subplotIdx)
        ax.plot(elapsedMsArr, quatDeviationArr, color='olive', label='quatDeviation')
        ax.legend()
        ax.set_title('Attitude deviation') 
        subplotIdx += 1

        ## Dt history 
        ax = fig.add_subplot(nRows, nCols, subplotIdx)
        ax.plot(elapsedMsArr[1:-1], globalDtArr[1:-1], color='blue', label='simCnt dt')
        ax.legend()
        ax.set_title('Simulation ticks') 
        subplotIdx += 1

        ## d(simCnt/ctrlCnt)/dt history 
        ax = fig.add_subplot(nRows, nCols, subplotIdx)
        ax.plot(elapsedMsArr[1:-1], phyDivPidEdDtArr, color='red', label='d(simCnt/ctrlCnt)/dt')
        ax.legend()
        ax.set_title('Tick ratio') 
        subplotIdx += 1

        ## Maxcv history 
        ax = fig.add_subplot(nRows, nCols, subplotIdx)
        ax.plot(elapsedMsArr, maxcvArr, color='blue')
        ax.legend()
        ax.set_title('Constraint Violation of IK Estimator') 
        subplotIdx += 1

        fig.tight_layout()
        #plt.show(block=True)
        plt.savefig(f'{CWD}/episode_stats/{figName}')

        rmseM = error_stats.rmse(massArr, plannedMassArr)
        rmseMDot = error_stats.rmse(massDtArr, plannedMassDtArr)
        rmseR = error_stats.rmse(posArr, plannedPosArr)
        rmseRDot = error_stats.rmse(velArr, plannedVelArr)
        rmseQ = error_stats.rmse([numpy.array([q.getR(), q.getI(), q.getJ(), q.getK()]) for q in quatArr], [numpy.array([1., 0., 0., 0.]) for i in quatArr])

        tfMeasuredTot += tfMeasured
        tfGuidanceTot += tfGuidance
        rmseMTot += rmseM 
        rmseMDotTot += rmseMDot 
        rmseRTot += rmseR
        rmseRDotTot += rmseRDot
        rmseQTot += rmseQ

        # Reference https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html?highlight=pearson
        badSchedulingArr = [(1+f if f > 0 else 0) for f in phyDivPidEdDtArr]
        pearsonrTickAndQuatErr = pearsonr(quatDeviationArr[1:-1], badSchedulingArr) 
        if not numpy.isnan(pearsonrTickAndQuatErr[0]):        
            pearsonrTickAndQuatErrTot += pearsonrTickAndQuatErr[0] 
            pearsonrTickAndQuatErrTotNep += 1

        pearsonrTickAndMdotErr = pearsonr([massDtArr[i]/plannedMassDtArr[i] for i,_ in enumerate(massDtArr)], badSchedulingArr) 
        if not numpy.isnan(pearsonrTickAndMdotErr[0]):        
            pearsonrTickAndMdotErrTot += pearsonrTickAndMdotErr[0] 
            pearsonrTickAndMdotErrTotNep += 1

        pearsonrMxCvAndQuatErr = pearsonr(quatDeviationArr, maxcvArr) 
        if not numpy.isnan(pearsonrMxCvAndQuatErr[0]):        
            pearsonrMxCvAndQuatErrTot += pearsonrMxCvAndQuatErr[0] 
            pearsonrMxCvAndQuatErrTotNep += 1

        pearsonrMxCvAndMdotErr = pearsonr([massDtArr[i]/plannedMassDtArr[i] for i,_ in enumerate(massDtArr)], maxcvArr[1:-1]) 
        if not numpy.isnan(pearsonrMxCvAndMdotErr[0]):        
            pearsonrMxCvAndMdotErrTot += pearsonrMxCvAndMdotErr[0] 
            pearsonrMxCvAndMdotErrTotNep += 1

        pearsonrMxCvAndTickRatio = pearsonr(badSchedulingArr, maxcvArr[1:-1]) 
        if not numpy.isnan(pearsonrMxCvAndTickRatio[0]):        
            pearsonrMxCvAndTickRatioTot += pearsonrMxCvAndTickRatio[0] 
            pearsonrMxCvAndTickRatioTotNep += 1

        pearsonrFile.write(f'episode#{nowEp+1:02d}, pearsonrTickAndQuatErr.r={pearsonrTickAndQuatErr[0]:+03.6f}, pearsonrTickAndMdotErr.r={pearsonrTickAndMdotErr[0]:+03.6f}, pearsonrMxCvAndQuatErr.r={pearsonrMxCvAndQuatErr[0]:+03.6f}, pearsonrMxCvAndMdotErr.r={pearsonrMxCvAndMdotErr[0]:+03.6f}, pearsonrMxCvAndTickRatio.r={pearsonrMxCvAndTickRatio[0]:+03.6f}\n')
        
        nowEp += 1

avgTfMeasured = tfMeasuredTot/nowEp
avgTfMeasured = format(avgTfMeasured, '.2f')
avgTfGuidance = tfGuidanceTot/nowEp
avgTfGuidance = format(avgTfGuidance, '.2f')

avgRmseM = rmseMTot/nowEp
avgRmseM = format(avgRmseM, '.2f')

avgRmseMDot = rmseMDotTot/nowEp-2
avgRmseMDot = format(avgRmseMDot, '.2f')

avgRmseR = rmseRTot/nowEp
avgRmseR = format(avgRmseR, '.2f')

avgRmseRDot = rmseRDotTot/nowEp
avgRmseRDot = format(avgRmseRDot, '.2f')

avgRmseQ = rmseQTot/nowEp
avgRmseQ = format(avgRmseQ, '.7f')

avgTickQuat = pearsonrTickAndQuatErrTot/pearsonrTickAndQuatErrTotNep
avgTickQuat = format(avgTickQuat, '.2f')
avgTickMdot = pearsonrTickAndMdotErrTot/pearsonrTickAndMdotErrTotNep
avgTickMdot = format(avgTickMdot, '.2f')
avgTickCv = pearsonrMxCvAndTickRatioTot/pearsonrMxCvAndTickRatioTotNep
avgTickCv = format(avgTickCv, '.2f')
avgCvQuat = pearsonrMxCvAndQuatErrTot/pearsonrMxCvAndQuatErrTotNep
avgCvQuat = format(avgCvQuat, '.2f')
avgCvMdot = pearsonrMxCvAndMdotErrTot/pearsonrMxCvAndMdotErrTotNep
avgCvMdot = format(avgCvMdot, '.2f')

toPrint = []
toPrint.append(f'Averages of totally {nowEp} episodes')
toPrint.append(f'tfGuidance :   {avgTfGuidance}')
toPrint.append(f'tfMeasured :   {avgTfMeasured}')
toPrint.append(f'rmseR      :   {avgRmseR}')
toPrint.append(f'rmseRDot   :   {avgRmseRDot}')
toPrint.append(f'rmseQ      :   {avgRmseQ}')
toPrint.append(f'rmseM      :   {avgRmseM}')
toPrint.append(f'rmseMDot   :   {avgRmseMDot}')
toPrint.append(f'tickQuat   :   {avgTickQuat}')
toPrint.append(f'tickMdot   :   {avgTickMdot}')
toPrint.append(f'tickCv     :   {avgTickCv}')
toPrint.append(f'cvQuat     :   {avgCvQuat}')
toPrint.append(f'cvMdot     :   {avgCvMdot}')
rootLogger.info('\n'.join(toPrint))

with open(f'{dirpath}/pearsonr_all.txt', 'a+') as pearsonrFile: 
    pearsonrFile.write('\n'.join(toPrint))
