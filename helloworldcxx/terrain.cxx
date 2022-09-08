#include "terrain.h"
#include "filename.h" // Reference https://www.panda3d.org/reference/cxx/classFilename.html 
#include "pnmImage.h" // Reference https://www.panda3d.org/reference/cxx/classPNMImage.html
#include "bulletHeightfieldShape.h" // Reference https://www.panda3d.org/reference/cxx/classBulletHeightfieldShape.html
#include "bitMask.h"

Terrain::Terrain(BulletWorld* world, NodePath worldNp, NodePath camera) {
    heightScale = 10.0; // Resizes the heights, see "terrain.getElevation(x, y)" for points in (0, 0) ~ (256, 256) to see that the original height values are quite small and NOT DISTINGUISHABLE!
    // Usage reference https://docs.panda3d.org/1.10/cpp/programming/physics/bullet/collision-shapes?highlight=bulletheightfieldshape#heightfield-shape
    Filename heightmapImgName("assets/heightmap.png"); // this is a 256x256 image, not a "power-of-two-plus-one size" one
    PNMImage heightmapImg(heightmapImgName); // an on-stack var
    PT(BulletHeightfieldShape) htFldShape = new BulletHeightfieldShape(heightmapImg, heightScale); 
    htFldShape->set_use_diamond_subdivision(false);
    PT(BulletRigidBodyNode) htFldNode = new BulletRigidBodyNode("myterrainhtfld");
    htFldNode->add_shape(htFldShape);
    NodePath np = worldNp.attach_new_node(htFldNode);
    np.set_pos(0, 0, 0);
    np.set_collide_mask(BitMask32(0xffff)); // all on
    world->attach(htFldNode);

    geoMipIns = new GeoMipTerrain("myterrain"); 
    geoMipIns->set_heightfield(heightmapImgName); // Do not use "set_heightfield(PNMImage)" here, otherwise a strict "power-of-two-plus-one size" constraint would be imposed! By using "set_heightfield(Filename)", GeoMipTerrain can automatically scale the image into a heightfield compatible size. 
    geoMipIns->set_color_map(Filename("assets/heightmapcolor.jpg"));
     
    geoMipIns->set_block_size(32);
    geoMipIns->set_near_far(50.0, 100.0);
    geoMipIns->set_focal_point(camera);
      
    rootNp = geoMipIns->get_root();
    rootNp.set_sz(heightScale);
    rootNp.reparent_to(np);

    offset = heightmapImg.get_x_size() * 0.5;
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
