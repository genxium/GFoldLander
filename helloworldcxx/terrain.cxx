#include "terrain.h"
#include "filename.h" // Reference https://www.panda3d.org/reference/cxx/classFilename.html 
#include "pnmImage.h" // Reference https://www.panda3d.org/reference/cxx/classPNMImage.html
#include "bulletHeightfieldShape.h" // Reference https://www.panda3d.org/reference/cxx/classBulletHeightfieldShape.html

Terrain::Terrain(BulletWorld* world, NodePath* pWorldNp, NodePath* pCamera) {
    heightScale = 10.0; // Resizes the heights, see "terrain.getElevation(x, y)" for points in (0, 0) ~ (256, 256) to see that the original height values are quite small and NOT DISTINGUISHABLE!
    // Usage reference https://docs.panda3d.org/1.10/cpp/programming/physics/bullet/collision-shapes?highlight=bulletheightfieldshape#heightfield-shape
    PNMImage heightmapImg(Filename("assets/heightmap.png")); // [WARNING] deliberately used as an on-stack-variable
    PT(BulletHeightfieldShape) htFldShape = new BulletHeightfieldShape(heightmapImg, heightScale); 
    htFldShape->set_use_diamond_subdivision(false);
    PT(BulletRigidBodyNode) htFldNode = new BulletRigidBodyNode("Heightfield");
    htFldNode->add_shape(htFldShape);
    NodePath np = pWorldNp->attach_new_node(htFldNode);

    geoMipIns = new GeoMipTerrain("myterrain"); 
    geoMipIns->set_heightfield(heightmapImg);
    geoMipIns->set_color_map(Filename("assets/heightmapcolor.jpg"));
     
    geoMipIns->set_block_size(32);
    geoMipIns->set_near_far(50.0, 100.0);
    geoMipIns->set_focal_point(*pCamera);
      
    rootNp = geoMipIns->get_root();
    rootNp.set_sz(heightScale);
    rootNp.reparent_to(np);

    offset = heightmapImg.get_x_size() * 0.5 - 0.5;
    rootNp.set_pos(-offset, -offset, -heightScale*0.5); // To align with the collision shape
    geoMipIns->generate();
}

Terrain::~Terrain() {
    if (geoMipIns) {
        geoMipIns->clear_color_map();
    }
}

double Terrain::calcRectifiedZ(double x, double y) {
    return 0;
}