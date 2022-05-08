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

from panda3d.core import Vec3
import math

SEA_LEVEL_OFFSET = 20.0

class Wind:
	def __init__(self, profileId=1):
		"""
		0 --- Calm	less than 1 mph (0 m/s)	Smoke rises vertically
		1 --- Light air	1 - 3 mph
		0.5-1.5 m/s	Smoke drifts with air, weather vanes inactive
		2 --- Light breeze	4 - 7 mph
		2-3 m/s	Weather vanes active, wind felt on face, leaves rustle
		3 --- Gentle breeze	8 - 12 mph
		3.5-5 m/s	Leaves & small twigs move, light flags extend
		4 --- Moderate breeze	13 - 18 mph
		5.5-8 m/s	Small branches sway, dust & loose paper blows about
		5 --- Fresh breeze	19 - 24 mph
		8.5-10.5 m/s	Small trees sway, waves break on inland waters
		6 --- Strong breeze	25 - 31 mph
		11-13.5 m/s	Large branches sway, umbrellas difficult to use
		7 --- Moderate gale	32 - 38 mph
		14-16.5 m/s	Whole trees sway, difficult to walk against wind
		8 --- Fresh gale	39 - 46 mph
		17-20 m/s	Twigs broken off trees, walking against wind very difficult
		9 --- Strong gale	47 - 54 mph
		20.5-23.5 m/s	Slight damage to buildings, shingles blown off roof
		10 -- Whole gale	55 - 63 mph
		24-27.5 m/s	Trees uprooted, considerable damage to buildings
		11 -- Storm	64 - 73 mph
		28-31.5 m/s	Widespread damage, very rare occurrence
		12 -- Hurricane	over 73 mph
		over 32 m/s	Violent destruction
		"""
		self.profileId = profileId

	def airDensity(self, landerIns):
		rho0 = 1.225 # in kg/m^3
		if 1 == self.profileId:
			return rho0
		if 2 == self.profileId:
			return rho0
		elif 3 == self.profileId:
			# Reference https://physics.stackexchange.com/questions/299907/air-density-as-a-function-of-altitude-only
			T0 = 288.16 # in Kelvin 
			alphaT = 0.0065 # in Kelvin/m
			nAir = 5.2561
			h = landerIns.physicsContainerNP.getPos().getZ()
			Th = T0 - alphaT*h
			return rho0*math.pow(Th/T0, nAir-1) 
		else:
			return 0.0

	def vel(self, landerIns):
		v0 = Vec3(3.5, 0.0, 0.0)
		h0 = 5 # set the reference height as 5 meters above sea level
		h = landerIns.physicsContainerNP.getPos().getZ()
		if 1 == self.profileId:
			return Vec3(0.0, 0.0, 0.0)
		if 2 == self.profileId:
			return v0
		elif 3 == self.profileId:
			# Reference https://www.quora.com/What-is-the-average-wind-speed-at-different-altitudes?share=1
			return v0*(math.log(h+SEA_LEVEL_OFFSET)/math.log(h0+SEA_LEVEL_OFFSET)) 
		else:
			return Vec3(0.0, 0.0, 0.0)

	def applyAeroForceToFaces(self, landerIns):
		# We deemed that each vertice of a face has the same "rocketVel", which might not be true if the rocket is rotating (in any dir), yet during a controlled flight the rocket should be only rotating with infinitesimal angular velocity
		rocketVel = landerIns.physicsContainerNP.node().get_linear_velocity()
		relVel = (rocketVel-self.vel(landerIns)) # a.k.a. the "wind frame" velocity
		relVelN = relVel.normalized()
		relSpd = relVel.length()
		relSpd2 = relVel.lengthSquared()

		nowQuat = landerIns.physicsContainerNP.getQuat()
		for i, face in enumerate(landerIns.simBodyFaces):
			forceInWorld = self._calcForceOnFace(landerIns, nowQuat, face, landerIns.simBodyVertices, relVel, relVelN, relSpd, relSpd2)
			forcePosInBody = (landerIns.simBodyVertices[face[0]] + landerIns.simBodyVertices[face[1]] + landerIns.simBodyVertices[face[2]])/3
			forceOffsetWrtNodeInWorld = nowQuat.xform(forcePosInBody)

			landerIns.physicsContainerNP.node().applyForce(forceInWorld, forceOffsetWrtNodeInWorld)

		for j, legFaces in enumerate(landerIns.simLegFacesList):
			vertices = landerIns.simLegVerticesList[j]
			for i, face in enumerate(legFaces):
				forceInWorld = self._calcForceOnFace(landerIns, nowQuat, face, vertices, relVel, relVelN, relSpd, relSpd2)
				forcePosInBody = (vertices[face[0]] + vertices[face[1]] + vertices[face[2]])/3
				forceOffsetWrtNodeInWorld = nowQuat.xform(forcePosInBody)

				landerIns.physicsContainerNP.node().applyForce(forceInWorld, forceOffsetWrtNodeInWorld)

	def _calcForceOnFace(self, landerIns, nowQuat, face, vertices, relVel, relVelN, relSpd, relSpd2):
		# Reference https://github.com/bulletphysics/bullet3/blob/master/src/BulletSoftBody/btSoftBody.cpp
		fLift = Vec3(0., 0., 0.)
		facePerp = nowQuat.xform((vertices[face[1]]-vertices[face[0]]).cross(vertices[face[2]]-vertices[face[0]])) # vertices[] are in body frame, it should be noticed that each actual vertice position in body frame should be "vertices[i]+rocketPartPosWrtCOG" instead, but by subtraction the resulting "facePerp" would be the same
		faceNorm = facePerp.normalized()
		# Rectify "faceNorm" because it might be flipped when imported automatically
		faceNorm = faceNorm*(-1.0 if (faceNorm.dot(relVel) < 0) else +1.0)
		faceArea = facePerp.length()*0.5
		
		nDotV = faceNorm.dot(relVelN)

		kLF = landerIns.kLF(nDotV, relVel)
		kDG = landerIns.kDG(nDotV, relVel)

		fDrag = (relVelN*(-1.0)*nDotV)*(faceArea*relSpd2*self.airDensity(landerIns)*kDG*0.5) # Deliberately put scalars at the end of expression

		"""
		# TODO 
		Assume that the "sigma = angle of attack", it can be seen that this "if condition" is only true for "-90d < sigma < -10d || 10d < sigma < 90d", where "nDotV == cos(sigma)" and "math.sqrt(1.0 - nDotV * nDotV) == |sin(sigma)|". 
		
		This formula for fLift neither respects near surface rotational flow effects, nor respects the ground effect. 
		"""
		# Check angle of attack
		# cos(10º) = 0.98480
		if 0 < nDotV and nDotV < 0.98480:
			fLift = (faceNorm.cross(relVelN).cross(relVelN))*(0.5 * kLF * self.airDensity(landerIns) * relSpd * faceArea * math.sqrt(1.0 - nDotV * nDotV))

		return (fLift + fDrag)

	def cleanup(self):
		return
