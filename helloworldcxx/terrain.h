#include "bulletWorld.h"
#include "nodePath.h"
#include "geoMipTerrain.h" // Reference https://www.panda3d.org/reference/cxx/classGeoMipTerrain.html

class Terrain {
public:
    double heightScale;
    double offset;

    // [WARNING] A "NodePath" instance is often NOT used as a pointer "NodePath*" to keep track of a heap-mem block, instead it contains a member variable that actually keeps track of a heap-mem block, e.g. "PandaNode* actualHeapMemPt = rootNp.node()".    
    NodePath rootNp;
    // [WARNING] A "GeoMipTerrain" instance is NOT a subclass of "class ReferenceCount", i.e. couldn't be used as PT(GeoMipTerrain).
    GeoMipTerrain* geoMipIns = nullptr;

    Terrain(BulletWorld* world, NodePath* pWorldNp, NodePath* pCamera);
    virtual ~Terrain();

    double calcRectifiedZ(double x, double y);
};
