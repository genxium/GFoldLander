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

# Get the list of all files and directories
dirpath = CWD + '/episode_stats'
 
nowEp = 0
tfMeasuredTot, tfGuidanceTot, rmseMTot, rmseRTot, rmseRDotTot, rmseQTot = 0., 0., 0., 0., 0., 0.
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
	#rootLogger.info(f'tfMeasured = {tfMeasured}')

	"""
	"""
	# Remove the None elements
	posArr = [f for i,f in enumerate(posArr) if elapsedMsArr[i] is not None]
	plannedPosArr = [f for i,f in enumerate(plannedPosArr) if elapsedMsArr[i] is not None]
	velArr = [f for i,f in enumerate(velArr) if elapsedMsArr[i] is not None]
	plannedVelArr = [f for i,f in enumerate(plannedVelArr) if elapsedMsArr[i] is not None]
	massArr = [f for i,f in enumerate(massArr) if elapsedMsArr[i] is not None]
	plannedMassArr = [f for i,f in enumerate(plannedMassArr) if elapsedMsArr[i] is not None]

	quatArr = [f for i,f in enumerate(quatArr) if elapsedMsArr[i] is not None]
	globalDtArr = [f for i,f in enumerate(globalDtArr) if elapsedMsArr[i] is not None]
	maxcvArr = [f for i,f in enumerate(maxcvArr) if elapsedMsArr[i] is not None]

	elapsedMsArr = [f for f in elapsedMsArr if f is not None] 

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
	fig.canvas.manager.set_window_title ('case_plot')
	nRows, nCols = 3, 2

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

	## Quat deviation history
	quatDeviationArr = [((f.getR()-1.0)**2 + f.getI()**2 + f.getJ()**2 + f.getK()**2) for i,f in enumerate(quatArr)]
	ax = fig.add_subplot(nRows, nCols, subplotIdx)
	ax.plot(elapsedMsArr, quatDeviationArr, color='olive', label='quatDeviation')
	ax.legend()
	ax.set_title('Attitude deviation') 
	subplotIdx += 1

	## Dt history 
	ax = fig.add_subplot(nRows, nCols, subplotIdx)
	ax.plot(elapsedMsArr, globalDtArr, color='blue', label='global dt')
	ax.legend()
	ax.set_title('Physics engine tick intervals') 
	subplotIdx += 1

	## Maxcv history, NOTE THAT this subplot has been observed to be closely related to that of "Quat deviation history" in terms of "coincidence of peaks" when there was considerable "Quat deviation"! 
	ax = fig.add_subplot(nRows, nCols, subplotIdx)
	ax.plot(elapsedMsArr, maxcvArr, color='blue')
	ax.legend()
	ax.set_title('Constraint Violation of IK Estimator') 
	subplotIdx += 1

	fig.tight_layout()
	plt.show(block=True)

	rmseM = error_stats.rmse(massArr, plannedMassArr)
	rmseR = error_stats.rmse(posArr, plannedPosArr)
	rmseRDot = error_stats.rmse(velArr, plannedVelArr)
	rmseQ = error_stats.rmse([numpy.array([q.getR(), q.getI(), q.getJ(), q.getK()]) for q in quatArr], [numpy.array([1., 0., 0., 0.]) for i in quatArr])

	tfMeasuredTot += tfMeasured
	tfGuidanceTot += tfGuidance
	rmseMTot += rmseM 
	rmseRTot += rmseR
	rmseRDotTot += rmseRDot
	rmseQTot += rmseQ

	nowEp += 1

avgTfMeasured = tfMeasuredTot/nowEp
avgTfMeasured = format(avgTfMeasured, '.2f')
avgTfGuidance = tfGuidanceTot/nowEp
avgTfGuidance = format(avgTfGuidance, '.2f')

avgRmseM = rmseMTot/nowEp
avgRmseM = format(avgRmseM, '.2f')

avgRmseR = rmseRTot/nowEp
avgRmseR = format(avgRmseR, '.2f')

avgRmseRDot = rmseRDotTot/nowEp
avgRmseRDot = format(avgRmseRDot, '.2f')

avgRmseQ = rmseQTot/nowEp
avgRmseQ = format(avgRmseQ, '.7f')

toPrint = []
toPrint.append(f'Averages of totally {nowEp} episodes')
toPrint.append(f'tfGuidance :	{avgTfGuidance}')
toPrint.append(f'tfMeasured :	{avgTfMeasured}')
toPrint.append(f'rmseM		:	{avgRmseM}')
toPrint.append(f'rmseR		:	{avgRmseR}')
toPrint.append(f'rmseRDot	:	{avgRmseRDot}')
toPrint.append(f'rmseQ		:	{avgRmseQ}')

rootLogger.info('\n'.join(toPrint))
