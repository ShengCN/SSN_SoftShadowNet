#pragma once
#include <common.h>
#include <Render_DS/Render_DS.h>

/* Compact scene representations */
struct scene {
	cuda_container<glm::vec3> world_verts;
	cuda_type<AABB> world_AABB;
	cuda_type<glm::vec3> render_target_center;
	cuda_type<plane> ground_plane;
	cuda_container<glm::vec3> rd_samples;

	scene(std::vector<glm::vec3> &verts, 
	AABB &world_aabb, 
	std::vector<glm::vec3> &samples,
	vec3 center, 
	plane &ground):
	world_verts(verts),
	world_AABB(world_aabb), 
	rd_samples(samples),
	ground_plane(ground),
	render_target_center(center) {
	}
};

/* Rendering params shared by all objects */
struct render_param {
	ppc cur_ppc; 		/* render camera */
	vec3 light_pos, render_target_center;

	int ibl_h, ibl_w, patch_size;
	int camera_pitch, target_rot;
};

struct output_param {
	bool resume, verbose, base_avg;
	std::string output_folder;
	std::string ofname;
	image img;
};

struct pixel_pos {
	int x, y;

	std::string to_string() {
		std::ostringstream oss;
		oss << x << "," << y;
		return oss.str();
	}
};

void render_data(const exp_params &params);

void mask_render(scene &cur_scene, render_param &cur_rp, output_param &out);
void normal_render(scene &cur_scene, render_param &cur_rp, output_param &out);
void depth_render(scene &cur_scene, render_param &cur_rp, output_param &out);
void shadow_render(scene &cur_scene, render_param &cur_rp, output_param &out);
void touch_render(scene &cur_scene, render_param &cur_rp, output_param &out);

