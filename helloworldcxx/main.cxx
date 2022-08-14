#include "pandaFramework.h"
#include "windowFramework.h"
#include "nodePath.h"
#include "clockObject.h"

#include "asyncTask.h"
#include "genericAsyncTask.h"

#include "bulletWorld.h"
#include "bulletPlaneShape.h"
#include "bulletBoxShape.h"

#include "ambientLight.h"
#include "directionalLight.h"
#include "pointLight.h"
#include "spotlight.h"

// The global task manager
PT(AsyncTaskManager) taskMgr = AsyncTaskManager::get_global_ptr();
// The global clock
PT(ClockObject) globalClock = ClockObject::get_global_clock();
// Here's what we'll store the camera in.
NodePath camera;

BulletWorld *get_physics_world() {
	// physics_world is supposed to be an global variable,
	// but declaring global variables is not cool
	// for good programmers lol, instead, should use static keyword.
	static PT(BulletWorld) physics_world = new BulletWorld();
	return physics_world.p();
}

AsyncTask::DoneStatus update_scene(GenericAsyncTask* task, void* data) {
	// Get dt (from Python example) and apply to do_physics(float, int, int);
	ClockObject *co = ClockObject::get_global_clock();
	get_physics_world()->do_physics(co->get_dt(), 10, 1.0 / 180.0);

	return AsyncTask::DS_cont;
}

AsyncTask::DoneStatus spinCameraTask(GenericAsyncTask *task, void *data) {
	// Calculate the new position and orientation (inefficient - change me!)
	double time = globalClock->get_real_time();
	double angledegrees = time * 6.0;
	double angleradians = angledegrees * (3.14 / 180.0);
	LVecBase3 cameraOffset = LVecBase3(50*sin(angleradians), -50.0*cos(angleradians), 20);
	NodePath np_box = *((NodePath*)data);
	camera.set_pos(np_box.get_pos() + cameraOffset);
	camera.look_at(np_box);

	// Tell the task manager to continue this task the next frame.
	return AsyncTask::DS_cont;
}

int main(int argc, char *argv[]) {
	// All variables.
	PandaFramework framework;
	WindowFramework *window;

	// Init everything :D
	framework.open_framework(argc, argv);
	framework.set_window_title("Bullet Physics");

	window = framework.open_window();
	window->enable_keyboard();
	window->setup_trackball();

	// Light
	// [WARNING] without any lighting "assets/rocket.bam" would look totally white, because a "colored material" would only show its color when illuminated. In contrast, a "textured material" can show its color even if not illuminated, but it's not a common feature in CAD software.
	PT(DirectionalLight) d_light;
	d_light = new DirectionalLight("d_light"); 
	d_light->set_direction(LVecBase3(1, 1, -1));
	d_light->set_color(LColor(0.7, 0.7, 0.7, 1));
	NodePath dlnp = window->get_render().attach_new_node(d_light);

	window->get_render().clear_light();
	window->get_render().set_light(dlnp);

	camera = window->get_camera_group();
	taskMgr = AsyncTaskManager::get_global_ptr();

	// Make physics simulation.
	// Static world stuff.
	get_physics_world()->set_gravity(0, 0, -9.81f);

	PT(BulletPlaneShape) floor_shape = new BulletPlaneShape(LVecBase3(0, 0, 1), 1);
	PT(BulletRigidBodyNode) floor_rigid_node = new BulletRigidBodyNode("Ground");

	floor_rigid_node->add_shape(floor_shape);

	NodePath np_ground = window->get_render().attach_new_node(floor_rigid_node);
	np_ground.set_pos(0, 0, -2);
	get_physics_world()->attach(floor_rigid_node);

	// Dynamic world stuff.
	PT(BulletBoxShape) box_shape = new BulletBoxShape(LVecBase3(0.5, 0.5, 0.5));
	PT(BulletRigidBodyNode) box_rigid_node = new BulletRigidBodyNode("Box");

	box_rigid_node->set_mass(1.0f); // Gravity affects this rigid node.
	box_rigid_node->add_shape(box_shape);

	NodePath np_box = window->get_render().attach_new_node(box_rigid_node);
	NodePath np_box_model = window->load_model(framework.get_models(), "assets/rocket.bam");
	np_box_model.set_pos(0, 0, 0); // This is the positin within "np_box"
	np_box_model.reparent_to(np_box);

	np_box.set_pos(0, 0, 5);
	get_physics_world()->attach(box_rigid_node);

	// If we specify custom data instead of NULL, it will be passed as the second argument
	// to the task function.
	taskMgr->add(new GenericAsyncTask("Scene update", &update_scene, nullptr));
  	taskMgr->add(new GenericAsyncTask("Spins the camera", &spinCameraTask, &np_box));

	framework.main_loop();
	framework.close_framework();

	return 0;
}
