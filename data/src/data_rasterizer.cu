
#include "data_rasterizer.h"

__global__
void raster_hard_shadow(plane* grond_plane, 
						glm::vec3* world_verts, 
	                    int N,
	                    AABB* aabb,
	                    ppc cur_ppc,
	                    vec3 light_pos,
	                    glm::vec3* pixels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i_stride = blockDim.x * gridDim.x;

	for (int ind = idx; ind < cur_ppc._height * cur_ppc._width; ind += i_stride) {
		int i = ind - ind/cur_ppc._width * cur_ppc._width;
		int j = cur_ppc._height-1-(ind-i)/cur_ppc._width;

		int cur_ind = (cur_ppc._height - 1 - j) * cur_ppc._width + i;
		pixels[cur_ind] = vec3(0.0f);

		ray r; cur_ppc.get_ray(i, j, r.ro, r.rd);
		vec3 intersect_pos;
		if (!plane_ppc_intersect(grond_plane, r, intersect_pos)) {
			continue;
		}

		bool ret = false;
		r = {intersect_pos, glm::normalize(light_pos - intersect_pos)};
		ray_aabb_intersect(r, aabb, ret);
		if (ret) {
			ret = false;
			for (int ti = 0; ti < N / 3; ti += 1) {
				vec3 p0 = world_verts[3 * ti + 0];
				vec3 p1 = world_verts[3 * ti + 1];
				vec3 p2 = world_verts[3 * ti + 2];
				
				ray_triangle_intersect(r, p0, p1, p2, ret);
				if (ret) {
					pixels[cur_ind] = vec3(1.0f);
					break;
				}
			}
		}
	}
}

__global__
void raster_mask(glm::vec3* world_verts, int N, AABB* aabb, ppc cur_ppc, glm::vec3* pixels) {
	// use thread id as i, j
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int i_stride = blockDim.x * gridDim.x;
	for (int ind = idx; ind < cur_ppc._height * cur_ppc._width; ind += i_stride) {
		int i = ind - ind/cur_ppc._width * cur_ppc._width;
		int j = cur_ppc._height-1-(ind-i)/cur_ppc._width;

		int cur_ind = (cur_ppc._height - 1 - j) * cur_ppc._width + i;
		
		bool ret = false;
		ray r; cur_ppc.get_ray(i, j, r.ro, r.rd);
		ray_aabb_intersect(r, aabb, ret);
		if (ret) {
			ret = false;
			float min_tt = FLT_MAX;
			for (int ti = 0; ti < N / 3; ti += 1) {
				vec3 p0 = world_verts[3 * ti + 0];
				vec3 p1 = world_verts[3 * ti + 1];
				vec3 p2 = world_verts[3 * ti + 2];
				
				float t = ray_triangle_intersect(r, p0, p1, p2, ret);
				if (ret && t < min_tt) {
					min_tt = t;
					pixels[cur_ind] = vec3(1.0f);
				}
			}
		}
	}
}

__global__
void raster_normal(glm::vec3* world_verts, int N, AABB* aabb, ppc cur_ppc, glm::vec3* pixels) {
	// use thread id as i, j
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i_stride = blockDim.x * gridDim.x;

	for (int ind = idx; ind < cur_ppc._height * cur_ppc._width; ind += i_stride) {
		int i = ind - ind / cur_ppc._width * cur_ppc._width;
		int j = cur_ppc._height - 1 - (ind - i) / cur_ppc._width;

		int cur_ind = (cur_ppc._height - 1 - j) * cur_ppc._width + i;
		if (pixels[cur_ind].x > 0.1f) {
			continue;
		}

		bool ret = false;
		ray r; cur_ppc.get_ray(i, j, r.ro, r.rd);
		ray_aabb_intersect(r, aabb, ret);
		if (ret) {
			ret = false;
			float closest_z = FLT_MAX;
			for (int ti = 0; ti < N / 3; ti += 1) {
				vec3 p0 = world_verts[3 * ti + 0];
				vec3 p1 = world_verts[3 * ti + 1];
				vec3 p2 = world_verts[3 * ti + 2];

				float t = ray_triangle_intersect(r, p0, p1, p2, ret);
				if (ret && t < closest_z) {
					closest_z = t;
					vec3 n = glm::normalize(glm::cross(p1 - p0, p2 - p1));
					pixels[cur_ind] = n;
				}
			}
		}
	}
}

__global__
void raster_depth(glm::vec3* world_verts, int N, AABB* aabb, ppc cur_ppc, glm::vec3* pixels) {
	// use thread id as i, j
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i_stride = blockDim.x * gridDim.x;

	for (int ind = idx; ind < cur_ppc._height * cur_ppc._width; ind += i_stride) {
		int i = ind - ind/cur_ppc._width * cur_ppc._width;
		int j = cur_ppc._height-1-(ind-i)/cur_ppc._width;

		int cur_ind = (cur_ppc._height - 1 - j) * cur_ppc._width + i;
		if(pixels[cur_ind].x > 0.1f) {
			continue;
		}
		
		bool ret = false;
		ray r; cur_ppc.get_ray(i, j, r.ro, r.rd);
		ray_aabb_intersect(r, aabb, ret);
		if (ret) {
			ret = false;
			float closest_z = FLT_MAX;
			for (int ti = 0; ti < N / 3; ti += 1) {
				vec3 p0 = world_verts[3 * ti + 0];
				vec3 p1 = world_verts[3 * ti + 1];
				vec3 p2 = world_verts[3 * ti + 2];
				
				float z = ray_triangle_intersect(r, p0, p1, p2, ret);
				if (ret &&  z < closest_z) {
					closest_z = z;
					float value = (closest_z - cur_ppc._near)/(cur_ppc._far - cur_ppc._near);
					pixels[cur_ind] = vec3(value);
				}
			}
		}
	}
}

__global__ 
void raster_ground(plane *ground_plane, ppc cur_ppc, glm::vec3 *pixels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i_stride = blockDim.x * gridDim.x;

	for (int ind = idx; ind < cur_ppc._height * cur_ppc._width; ind += i_stride) { int i = ind - ind/cur_ppc._width * cur_ppc._width;
		int j = cur_ppc._height-1-(ind-i)/cur_ppc._width;
		int cur_ind = (cur_ppc._height - 1 - j) * cur_ppc._width + i;

		// compute the intersection point with the plane
		ray cur_ray; cur_ppc.get_ray(i, j, cur_ray.ro, cur_ray.rd);
		vec3 intersect_pos;
		if(plane_ppc_intersect(ground_plane, cur_ray, intersect_pos)) {
			pixels[cur_ind] = vec3(1.0f);
		} else {
			pixels[cur_ind] = vec3(0.0f);
		}
	}
}

__global__ 
void raster_height(glm::vec3 *world_verts, int N, AABB* aabb,plane *ground_plane, ppc cur_ppc, glm::vec3 *pixels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i_stride = blockDim.x * gridDim.x;

	for (int ind = idx; ind < cur_ppc._height * cur_ppc._width; ind += i_stride) { 
		int i = ind - ind/cur_ppc._width * cur_ppc._width;
		int j = cur_ppc._height-1-(ind-i)/cur_ppc._width;
		int cur_ind = (cur_ppc._height - 1 - j) * cur_ppc._width + i;

		// sky color 1.0
		pixels[cur_ind] = glm::vec3(1.0f);

		ray cur_ray; cur_ppc.get_ray(i, j, cur_ray.ro, cur_ray.rd);
		vec3 intersect_pos;
		if(plane_ppc_intersect(ground_plane, cur_ray, intersect_pos)) {
			pixels[cur_ind] = vec3(0.0f);
		} 		
		
		bool ret = false;
		ray r; cur_ppc.get_ray(i, j, r.ro, r.rd);
		ray_aabb_intersect(r, aabb, ret);
		if (ret) {
			ret = false;
			float closest_z = FLT_MAX;
			for (int ti = 0; ti < N / 3; ti += 1) {
				vec3 p0 = world_verts[3 * ti + 0];
				vec3 p1 = world_verts[3 * ti + 1];
				vec3 p2 = world_verts[3 * ti + 2];
				
				float z = ray_triangle_intersect(r, p0, p1, p2, ret);
				if (ret &&  z < closest_z) {
					vec3 p = r.ro + r.rd * z;
					float h = std::abs(glm::dot(p-ground_plane->p, ground_plane->n));
					pixels[cur_ind] = vec3(height_transform(h, 1.0f) * 0.9f);
				}
			}
		}
	}
}

__host__ __device__
float height_transform(float h, float norm_size) {
	return h/norm_size;	
}

__global__
void raster_touch(glm::vec3 *world_verts, int N, AABB* aabb,plane *ground_plane, ppc cur_ppc, vec3 *samples, int sn,glm::vec3 *pixels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i_stride = blockDim.x * gridDim.x;

	for (int ind = idx; ind < cur_ppc._height * cur_ppc._width; ind += i_stride) { 
		int i = ind - ind/cur_ppc._width * cur_ppc._width;
		int j = cur_ppc._height-1-(ind-i)/cur_ppc._width;
		int cur_ind = (cur_ppc._height - 1 - j) * cur_ppc._width + i;

		bool ret = false;
		ray r; cur_ppc.get_ray(i, j, r.ro, r.rd);
		vec3 intersect;
		if (!plane_ppc_intersect(ground_plane, r, intersect)) {
			continue;
		}
		
		float vis = 0.0f;
		for(int si = 0; si < sn; ++si) {
			r.rd = samples[si];
			r.ro = intersect;
			
			ret = false;
			ray_aabb_intersect(r, aabb, ret);
			if(!ret) {
				continue;
			}
			
			for (int ti = 0; ti < N / 3; ti += 1) {
				vec3 p0 = world_verts[3 * ti + 0];
				vec3 p1 = world_verts[3 * ti + 1];
				vec3 p2 = world_verts[3 * ti + 2];

				ret = false;
				ray_triangle_intersect(r, p0, p1, p2, ret);
				if(ret) {
					vis += 1;
					break;
				}
			}
		}
		vis = vis / (float)sn * 4.0 * pd::pi;
		
		if (vis > 1.0f) vis = 1.0f;
		vis = vis * vis * vis;

		// if (vis < 0.5f) vis = 0.0f;
		pixels[cur_ind] = vec3(vis);
	}
}