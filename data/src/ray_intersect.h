#pragma once
#include <common.h>
#include "ppc.h"
#include "mesh.h"

struct plane {
	glm::vec3 p, n;
};

__host__ __device__
bool plane_ppc_intersect(plane* ground_plane, ray& cur_ray, glm::vec3& intersection_pos);

__host__ __device__
float ray_triangle_intersect(const ray& r, vec3 p0, vec3 p1, vec3 p2, bool& ret);

__host__ __device__
void swap(float& a, float& b);

__host__ __device__
void ray_aabb_intersect(const ray&r, AABB* aabb, bool& ret);

