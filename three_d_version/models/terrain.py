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

from panda3d.core import BitMask32
from panda3d.bullet import ZUp
from panda3d.bullet import BulletHeightfieldShape
from panda3d.bullet import BulletRigidBodyNode

# Terrain required modules
from panda3d.core import Filename, PNMImage, GeoMipTerrain

class Terrain:
    def __init__(self, world, worldNP):
        # Ground with collidable terrain
        self.heightScale = 10.0 # Resizes the heights, see "terrain.getElevation(x, y)" for points in (0, 0) ~ (256, 256) to see that the original height values are quite small and NOT DISTINGUISHABLE!
        heightmapImg = PNMImage(Filename('models/heightmap.png')) # Note that this "png" image should be at least 16-bit in color space
        shape = BulletHeightfieldShape(heightmapImg, self.heightScale, ZUp)
        shape.setUseDiamondSubdivision(False)

        np = worldNP.attachNewNode(BulletRigidBodyNode('myterrainhtfld'))
        np.node().addShape(shape)
        np.setPos(0, 0, 0)
        np.setCollideMask(BitMask32.allOn())
        self.terrainRigidBodyNP = np 
        world.attachRigidBody(self.terrainRigidBodyNP.node())

        self.geoMipIns = GeoMipTerrain("myterrain")
        self.geoMipIns.setHeightfield('models/heightmap.png')
        self.geoMipIns.setColorMap("models/heightmapcolor.jpg")
        #self.geoMipIns.setBruteforce(True)

        self.geoMipIns.setBlockSize(32)
        self.geoMipIns.setNear(50)
        self.geoMipIns.setFar(100)
        self.geoMipIns.setFocalPoint(base.camera)

        self.rootNP = self.geoMipIns.getRoot()
        self.rootNP.setSz(self.heightScale)
        self.rootNP.reparentTo(self.terrainRigidBodyNP)

        self.offset = heightmapImg.getXSize()*0.5 - 0.5
        self.rootNP.setPos(-self.offset, -self.offset, -self.heightScale*0.5) # To align with the collision shape

        self.geoMipIns.generate()

    def calcRectifiedZ(self, x, y):
        return self.rootNP.getSz()*self.geoMipIns.getElevation(
                x+self.offset,
                y+self.offset
            ) - self.heightScale*0.5 # A similar offset is necessary just like that of "terrainRootNP"

    def cleanup(self):
        # Panda3D NodePath variables don't need manual deallocation, they'll be deallocated once detached from "worldNP"
        self.geoMipIns = None
