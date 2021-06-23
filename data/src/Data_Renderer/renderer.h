#pragma once
#include <common.h>
#include <Render_DS/Render_DS.h>

/* Rendering params shared by all objects */
struct render_param{
	int w, h;    		/* image space dimension */
	ppc cur_ppc; 		/* render camera */
	vec3 light_pos;     /* assume the light is point light */
	int camera_pitch, target_rot; /* Scene Variation */ 
}

/* Compact scene representations */
struct scene {
	std::vector<glm::vec3> world_verts;
	AABB world_AABB;
	glm::vec3 render_target_center;
	plane ground_plane;
}

struct pixel_pos {
	int x, y;

	std::string to_string() {
		std::ostringstream oss;
		oss << x << "," << y;
		return oss.str();
	}
};

void shadow_render(scene &cur_scene, render_param &cur_rp, image &out_img);
void normal_render(scene &cur_scene, render_param &cur_rp, image &out_img);
void depth_render(scene &cur_scene, render_param &cur_rp, image &out_img);
void ground_render(scene &cur_scene, render_param &cur_rp, image &out_img);
void touch_render(scene &cur_scene, render_param &cur_rp, image &out_img);

void shadow_render(glm::vec3* world_verts_cuda, 
	int N, 
	plane* ground_plane, 
	AABB* aabb_cuda, 
	ppc &cur_ppc, 
	vec3 render_target_center,
	int camera_pitch, 
	int target_rot, 
	glm::vec3* pixels, 
	glm::vec3* tmp_pixels,
	unsigned int* out_pixels, 
	image &out_img);

void mask_render(glm::vec3* world_verts_cuda, 
				int N, 
				AABB* aabb_cuda, 
				ppc &cur_ppc, 
				glm::vec3* pixels, 
				unsigned int* out_pixels, 
				image &out_img, 
				std::string output_fname);


void normal_render(glm::vec3* world_verts_cuda, 
				int N, 
				AABB* aabb_cuda, 
				ppc &cur_ppc, 
				glm::vec3* pixels, 
				unsigned int* out_pixels, 
				image &out_img, 
				std::string output_fname);


void depth_render(glm::vec3* world_verts_cuda, 
				int N, 
				AABB* aabb_cuda, 
				ppc &cur_ppc, 
				glm::vec3* pixels, 
				unsigned int* out_pixels, 
				image &out_img, 
				std::string output_fname);


void ground_render(plane* ground_plane,  
				ppc &cur_ppc, 
				glm::vec3* pixels, 
				unsigned int* out_pixels, 
				image &out_img, 
				std::string output_fname);

void touch_render(glm::vec3 *world_verts_cuda, 
				int N, 
				AABB* aabb,
				plane* ground_plane,  
				ppc &cur_ppc, 
				glm::vec3* pixels, 
				unsigned int* out_pixels, 
				image &out_img, 
				std::string output_fname);

void render_data(const std::string model_file, const std::string output_folder);