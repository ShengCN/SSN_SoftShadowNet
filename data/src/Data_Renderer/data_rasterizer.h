#pragma once
#include <common.h>
#include <Render_DS/Render_DS.h>
#include "ray_intersect.h"

__global__
void raster_hard_shadow(plane* grond_plane, 
						glm::vec3* world_verts, 
	                    int N,
	                    AABB* aabb,
	                    ppc cur_ppc,
	                    vec3 light_pos,
	                    glm::vec3* pixels);

__global__
void raster_mask(glm::vec3* world_verts, 
				 int N, 
				 AABB* aabb, 
				 ppc cur_ppc, 
				 glm::vec3* pixels);


__global__
void raster_normal(glm::vec3* world_verts, 
				int N, 
				AABB* aabb, 
				ppc cur_ppc, 
				glm::vec3* pixels);

__global__
void raster_depth(glm::vec3* world_verts, 
				int N, 
				AABB* aabb, 
				ppc cur_ppc, 
				glm::vec3* pixels);

__global__
void raster_touch(glm::vec3 *world_verts, 
				int N, 
				AABB *aabb,
				plane *ground_plane, 
				ppc cur_ppc, 
				vec3 *samples, 
				int sn,
				glm::vec3 *pixels);

__global__
void raster_height(glm::vec3 *world_verts, 
				int N, 
				AABB *aabb,
				plane *ground_plane, 
				ppc cur_ppc, 
				glm::vec3 *pixels);