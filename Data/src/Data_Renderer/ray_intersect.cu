#include "ray_intersect.h"

__host__ __device__
bool ray_plane_intersect(plane* ground_plane, ray& cur_ray, glm::vec3& intersection_pos) {
	if(glm::dot(cur_ray.rd, ground_plane->n) > 0.0f) {
		return false;
	}

	float t = glm::dot(ground_plane->p - cur_ray.ro, ground_plane->n) / glm::dot(cur_ray.rd, ground_plane->n);
	intersection_pos = cur_ray.ro + cur_ray.rd * t;
	return true;
}

__host__ __device__
float ray_triangle_intersect(const ray& r, vec3 p0, vec3 p1, vec3 p2, bool& ret) {
	glm::vec3 v0v1 = p1 - p0;
	glm::vec3 v0v2 = p2 - p0;
	glm::mat3 m;
	m[0] = -r.rd; m[1] = v0v1; m[2] = v0v2;
	glm::vec3 b = r.ro - p0;

	if (std::abs(glm::determinant(m)) < 1e-6f) {
		ret = false;
		return 0.0f;
	}

	glm::vec3 x = glm::inverse(m) * b;
	float t = x.x, u = x.y, v = x.z;
	if (t <= 0.0 || u < 0.0 || v < 0.0 || u > 1.0 || v >1.0 || u + v < 0.0 || u + v > 1.0) {
		ret = false; return 0.0f;
	}
	
	ret = true;
	return std::sqrt(glm::dot(r.rd * t, r.rd * t));
}

__host__ __device__
void swap(float& a, float& b) {
	float c = a;
	a = b;
	b = c;
}

__host__ __device__
void ray_aabb_intersect(const ray&r, AABB* aabb, bool& ret) {
	float tmin = (aabb->p0.x - r.ro.x) / r.rd.x;
	float tmax = (aabb->p1.x - r.ro.x) / r.rd.x;

	if (tmin > tmax) swap(tmin, tmax);

	float tymin = (aabb->p0.y - r.ro.y) / r.rd.y;
	float tymax = (aabb->p1.y - r.ro.y) / r.rd.y;

	if (tymin > tymax) swap(tymin, tymax);

	if ((tmin > tymax) || (tymin > tmax)) {
		ret = false;
		return;
	}

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (aabb->p0.z - r.ro.z) / r.rd.z;
	float tzmax = (aabb->p1.z - r.ro.z) / r.rd.z;

	if (tzmin > tzmax) swap(tzmin, tzmax);

	if ((tmin > tzmax) || (tzmin > tmax)) {
		ret = false;
		return; 
	}

	ret = true;
}