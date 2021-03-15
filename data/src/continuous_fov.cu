#include <iostream>
#include <glm/mat3x3.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
#include <limits>
#include <stdio.h>
#include <math.h>

#include "cxxopts.hpp"
#include "common.h"
#include "mesh.h"
#include "model_loader.h"
#include "ppc.h"

const int w = 256, h = 256;
using namespace purdue;
// int i_begin = 256 - (int)((60.0 / 90.0) * 256 * 0.5);
// int i_begin = 256 - 80;
int i_begin = 256/2;
int nx = 256, ny = 256;
int tx = 8, ty = 64;
int dev = 0;
bool resume = false;
bool camera_change = false;
bool verbose = true;
const int patch_size = 8;
int counter=0,total_counter=0;
float light_relative_length = 11.0f;
std::vector<float> cam_pitch, model_rot;
timer profiling;
std::string model_file, output_folder;
bool render_shadow, render_mask, render_normal, render_depth, render_ground, render_height;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

struct pixel_pos {
	int x, y;

	std::string to_string() {
		std::ostringstream oss;
		oss << x << "," << y;
		return oss.str();
	}
};

std::vector<pixel_pos> ibl_map;

bool load_mesh(const std::string fname, std::shared_ptr<mesh> &m) {
	auto loader = model_loader::create(fname);
	return loader->load_model(fname, m);
}

struct image {
	std::vector<unsigned int> pixels;
	int w, h;
	image(int w, int h) :w(w), h(h) {
		// pixels.resize(w * h);
		clear();
	}

	void clear() {
		pixels.resize(w * h, 0xffffffff);
	}

	void set_pixel(int u, int v, vec3 p) {
		size_t ind = (h - 1 - v) * w + u;
		reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[0] = (unsigned char)(255.0 * p.x);
		reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[1] = (unsigned char)(255.0 * p.y);
		reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[2] = (unsigned char)(255.0 * p.z);
		reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[3] = (unsigned char)(255);
	}

	glm::vec3 at(int u, int v) {
		size_t ind = (h - 1 - v) * w + u;
		unsigned int p = pixels.at(ind);
		glm::vec3 ret(reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[0]/255.0f, reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[1]/255.0f, reinterpret_cast<unsigned char*>(&(pixels.at(ind)))[2]/255.0);
		return ret;
	}

	bool save(const std::string fname) {
		return save_image(fname, pixels.data(), w, h);
	}
};

struct ray {
	vec3 ro, rd;
};

struct plane {
	vec3 p, n;
};

bool save(const std::string fname, unsigned int* pixels, int w, int h) {
    return save_image(fname, pixels, w, h);
}

__host__ __device__
bool plane_ppc_intersect(plane* ground_plane, ray& cur_ray, glm::vec3& intersection_pos) {
	if(glm::dot(cur_ray.rd, ground_plane->n) > 0.0f) {
		return false;
	}

	float t = glm::dot(ground_plane->p - cur_ray.ro, ground_plane->n) / glm::dot(cur_ray.rd, ground_plane->n);
	intersection_pos = cur_ray.ro + cur_ray.rd * t;
	return true;
}

// given an ibl map and light center position
// return the 3D light position
vec3 compute_light_pos(int x, int y, int w = 512, int h = 256) {
	//todo
	float x_fract = (float)x / w, y_fract = (float)y / h;
	deg alpha, beta;
	alpha = x_fract * 360.0f - 90.0f; beta = y_fract * 180.0f - 90.0f;
	return vec3(cos(deg2rad(beta)) * cos(deg2rad(alpha)), sin(deg2rad(beta)), cos(deg2rad(beta)) * sin(deg2rad(alpha)));
}

__host__ __device__
float ray_triangle_intersect(const ray& r, vec3 p0, vec3 p1, vec3 p2, bool& ret) {
	glm::vec3 v0v1 = p1 - p0;
	glm::vec3 v0v2 = p2 - p0;
	glm::mat3 m;
	m[0] = -r.rd; m[1] = v0v1; m[2] = v0v2;
	glm::vec3 b = r.ro - p0;

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

	//if (tzmin > tmin)
	//	tmin = tzmin;

	//if (tzmax < tmax)
	//	tmax = tzmax;

	ret = true;
}

__host__ __device__
void set_pixel(vec3 c, unsigned int& p) {
	reinterpret_cast<unsigned char*>(&p)[0] = (unsigned char)(255.0 * c.x);
	reinterpret_cast<unsigned char*>(&p)[1] = (unsigned char)(255.0 * c.y);
	reinterpret_cast<unsigned char*>(&p)[2] = (unsigned char)(255.0 * c.z);
	reinterpret_cast<unsigned char*>(&p)[3] = (unsigned char)(255);
}

__global__
void reset_pixel(const vec3 c, glm::vec3* array) {
	// use thread id as i, j
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int jdx = blockDim.y * blockIdx.y + threadIdx.y;

	int i_stride = blockDim.x * gridDim.x;
	int j_stride = blockDim.y * gridDim.y;

	// iterate over the output image
	for (int j = jdx; j < h; j += j_stride) 
		for (int i = idx; i < w; i += i_stride) {
			// set_pixel(pixel_value, pixels[(cur_ppc._height - 1 - j) * cur_ppc._width + i]);
            int ind = (h-1-j) * w + i;
            array[ind] = c;
		}
}

__global__
void to_unsigned_array(int w, int h, int patch_size, glm::vec3* array_a, unsigned int* array_out) {
	// use thread id as i, j
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int jdx = blockDim.y * blockIdx.y + threadIdx.y;

	int i_stride = blockDim.x * gridDim.x;
	int j_stride = blockDim.y * gridDim.y;
	float weight = 1.0f/(patch_size * patch_size);

	// iterate over the output image
	for (int j = jdx; j < h; j += j_stride) 
		for (int i = idx; i < w; i += i_stride) {
			// set_pixel(pixel_value, pixels[(cur_ppc._height - 1 - j) * cur_ppc._width + i]);
			int ind = (h-1-j) * w + i;
            set_pixel(array_a[ind] * weight, array_out[ind]);
		}
}

__global__
void add_array(int w, int h, glm::vec3* array_a, glm::vec3* array_out) {
	// use thread id as i, j
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int jdx = blockDim.y * blockIdx.y + threadIdx.y;

	int i_stride = blockDim.x * gridDim.x;
	int j_stride = blockDim.y * gridDim.y;

	// iterate over the output image
	for (int j = jdx; j < h; j += j_stride) 
		for (int i = idx; i < w; i += i_stride) {
			// set_pixel(pixel_value, pixels[(cur_ppc._height - 1 - j) * cur_ppc._width + i]);
            int ind = (h-1-j) * w + i;
            array_out[ind] += array_a[ind];
		}
}

__global__
void raster_hard_shadow(plane* grond_plane, 
						glm::vec3* world_verts, 
	                    int N,
	                    AABB* aabb,
	                    ppc cur_ppc,
	                    vec3 light_pos,
	                    glm::vec3* pixels) {
	// use thread id as i, j
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i_stride = blockDim.x * gridDim.x;

	for (int ind = idx; ind < cur_ppc._height * cur_ppc._width; ind += i_stride) {
		int i = ind - ind/cur_ppc._width * cur_ppc._width;
		int j = h-1-(ind-i)/cur_ppc._width;

		// compute the intersection point with the plane
		ray cur_ray; cur_ppc.get_ray(i, j, cur_ray.ro, cur_ray.rd);
		vec3 intersect_pos;
		if(!plane_ppc_intersect(grond_plane, cur_ray, intersect_pos)) {
			pixels[(cur_ppc._height - 1 - j)] = vec3(0.0f);
			continue;
		}

		bool ret = false;
		ray r = { intersect_pos, light_pos - intersect_pos };
		if (r.rd.y < 0.0f) {
			pixels[(cur_ppc._height - 1 - j) * cur_ppc._width + i] = vec3(0.0f);
			continue;
		}

		ray_aabb_intersect(r, aabb, ret);
		if (ret) {
			ret = false;
			for (int ti = 0; ti < N / 3; ti += 1) {
				vec3 p0 = world_verts[3 * ti + 0];
				vec3 p1 = world_verts[3 * ti + 1];
				vec3 p2 = world_verts[3 * ti + 2];
				
				float t = ray_triangle_intersect(r, p0, p1, p2, ret);
				if (ret) {
					pixels[(cur_ppc._height - 1 - j) * cur_ppc._width + i] = vec3(1.0f);
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
		int j = h-1-(ind-i)/cur_ppc._width;

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
		int j = h - 1 - (ind - i) / cur_ppc._width;

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
		int j = h-1-(ind-i)/cur_ppc._width;

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
		int j = h-1-(ind-i)/cur_ppc._width;
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

__host__ __device__
float height_transform(float h, float norm_size) {
	return h/norm_size;	
}

__global__ 
void raster_height(glm::vec3 *world_verts, int N, AABB* aabb,plane *ground_plane, ppc cur_ppc, glm::vec3 *pixels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i_stride = blockDim.x * gridDim.x;

	for (int ind = idx; ind < cur_ppc._height * cur_ppc._width; ind += i_stride) { 
		int i = ind - ind/cur_ppc._width * cur_ppc._width;
		int j = h-1-(ind-i)/cur_ppc._width;
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
	image &out_img) {
	// ----------------------------------------------- SHADOW BASES ----------------------------------------------------
	// 8x8 patch rendering 
	// [0:512]x[i_begin:256] -> [0:128] x [i_begin/8:64]
	for(int ibl_i = 0; ibl_i <= 512/patch_size; ++ibl_i) {
		for(int ibl_j = i_begin/patch_size; ibl_j <= 256/patch_size; ++ibl_j) {
			pixel_pos begin_pixel_pos = {ibl_i * patch_size, ibl_j * patch_size};
			vec3 light_position = compute_light_pos(begin_pixel_pos.x, begin_pixel_pos.y) * light_relative_length + render_target_center;
			
			// std::cerr << ibl_i << " " << ibl_j << " " << to_string(light_position) << std::endl;

			// char buff[100];
			// snprintf(buff, sizeof(buff), "pitch_%d_rot_%d_ibli_%d_iblj_%d", (int)camera_pitch, (int)target_rot, begin_pixel_pos.x, begin_pixel_pos.y);
			// std::string cur_prefix = buff;

			std::string cur_prefix;
			float fov = cur_ppc.get_fov();
			char buff[100]; snprintf(buff, sizeof(buff), "pitch_%d_rot_%d_fov_%d_ibli_%d_iblj_%d", (int)camera_pitch, (int)target_rot, (int)fov,begin_pixel_pos.x, begin_pixel_pos.y);
			cur_prefix = buff;
			
			std::string output_fname = output_folder + "/" + cur_prefix + "_shadow.png";
			
			if(resume) {
				if(exists_test(output_fname)) {
					std::cout << output_fname << " is skipped \n";
					++counter;
					continue;
				}
			}
			
			profiling.tic();
			int grid = 512, block = (w * h + grid - 1)/grid;
			reset_pixel<<<grid, block>>>(vec3(0.0f), pixels);
			gpuErrchk(cudaDeviceSynchronize());
			for(int patch_i = 0; patch_i < patch_size; ++patch_i)
				for(int patch_j = 0; patch_j < patch_size; ++patch_j) {
					pixel_pos light_pixel_pos = {begin_pixel_pos.x + patch_i, begin_pixel_pos.y + patch_j};
					vec3 light_position = compute_light_pos(light_pixel_pos.x, light_pixel_pos.y) * light_relative_length + render_target_center;
					
					reset_pixel<<<grid, block>>>(vec3(0.0f), tmp_pixels);
					gpuErrchk(cudaDeviceSynchronize());

					raster_hard_shadow<<<grid,block>>>(ground_plane, 
						world_verts_cuda, 
						N, 
						aabb_cuda, 
						cur_ppc, 
						light_position, 
						tmp_pixels);
					gpuErrchk(cudaPeekAtLastError());
					gpuErrchk(cudaDeviceSynchronize());
					
					add_array<<<grid, block>>>(w, h, tmp_pixels, pixels);
					gpuErrchk(cudaDeviceSynchronize());
			}
			to_unsigned_array<<<grid, block>>>(w, h, patch_size, pixels, out_pixels);
			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
			
			profiling.toc();
			if (verbose) {
				std::string total_time = profiling.to_string();
				std::cerr << fov << "," << begin_pixel_pos.x << "," << begin_pixel_pos.y << " shadow rendering time: " << total_time << std::endl;
			}

			out_img.save(output_fname);
			std::cerr << "Finish: " << (float)++counter / total_counter * 100.0f << "% \r";
		}
	}
	// ----------------------------------------------- SHADOW BASES ----------------------------------------------------
}

void mask_render(glm::vec3* world_verts_cuda, int N, AABB* aabb_cuda, ppc &cur_ppc, glm::vec3* pixels, unsigned int* out_pixels, image &out_img, std::string output_fname) {
	// ----------------------------------------------- Masks ----------------------------------------------------
	int grid = 512, block = 32 * 4;

	timer profiling;
	profiling.tic();
	reset_pixel <<<grid, block >> > (vec3(0.0f), pixels);
	gpuErrchk(cudaDeviceSynchronize());
	raster_mask << <grid, block >> > (world_verts_cuda,
		N,
		aabb_cuda,
		cur_ppc,
		pixels);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	to_unsigned_array << <grid, block >> > (w, h, 1, pixels, out_pixels);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	profiling.toc();
	if (verbose) {
		std::string total_time = profiling.to_string();
		std::cerr << "mask total time: " << total_time << std::endl;
	}

	if (output_fname != "")
		out_img.save(output_fname);
	// ----------------------------------------------- Masks ----------------------------------------------------
}

void normal_render(glm::vec3* world_verts_cuda, int N, AABB* aabb_cuda, ppc &cur_ppc, glm::vec3* pixels, unsigned int* out_pixels, image &out_img, std::string output_fname) {
	// ----------------------------------------------- Normal ----------------------------------------------------

	//if(resume) {
	//	if(exists_test(output_fname)) {
	//		std::cout << output_fname << " is skipped \n";
	//		continue;
	//	}
	//}
	int grid = 512, block = 32 * 4;

	timer profiling;
	profiling.tic();
	reset_pixel << <grid, block >> > (vec3(0.0f), pixels);
	gpuErrchk(cudaDeviceSynchronize());
	raster_normal << <grid, block >> > (world_verts_cuda,
		N,
		aabb_cuda,
		cur_ppc,
		pixels);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	to_unsigned_array << <grid, block >> > (w, h, 1, pixels, out_pixels);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	profiling.toc();
	if (verbose) {
		std::string total_time = profiling.to_string();
		std::cerr << "normal total time: " << total_time << std::endl;
	}
	out_img.save(output_fname);
	// ----------------------------------------------- Normal ----------------------------------------------------
}


void depth_render(glm::vec3* world_verts_cuda, int N, AABB* aabb_cuda, ppc &cur_ppc, glm::vec3* pixels, unsigned int* out_pixels, image &out_img, std::string output_fname) {
	// ----------------------------------------------- Depth ----------------------------------------------------

	//if(resume) {
	//	if(exists_test(output_fname)) {
	//		std::cout << output_fname << " is skipped \n";
	//		continue;
	//	}
	//}
	int grid = 512, block = 32 * 4;

	timer profiling;
	profiling.tic();
	reset_pixel << <grid, block >> > (vec3(0.0f), pixels);
	gpuErrchk(cudaDeviceSynchronize());
	raster_depth << <grid, block >> > (world_verts_cuda,
		N,
		aabb_cuda,
		cur_ppc,
		pixels);
	gpuErrchk(cudaPeekAtLastError());	
	gpuErrchk(cudaDeviceSynchronize());

	to_unsigned_array << <grid, block >> > (w, h, 1, pixels, out_pixels);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	profiling.toc();
	if (verbose) {
		std::string total_time = profiling.to_string();
		std::cerr << "depth total time: " << total_time << std::endl;
	}
	out_img.save(output_fname);
	// ----------------------------------------------- Depth ----------------------------------------------------
}

void ground_render(plane* ground_plane,  ppc &cur_ppc, glm::vec3* pixels, unsigned int* out_pixels, image &out_img, std::string output_fname) {
	int grid = 512, block = 32 * 4;
	timer profiling;
	profiling.tic();
	reset_pixel<<<grid, block>>> (vec3(0.0f), pixels);
	gpuErrchk(cudaDeviceSynchronize());
	
	raster_ground << <grid, block>> > (ground_plane, cur_ppc, pixels);
	
	gpuErrchk(cudaPeekAtLastError());	
	gpuErrchk(cudaDeviceSynchronize());

	to_unsigned_array << <grid, block >> > (w, h, 1, pixels, out_pixels);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	profiling.toc();
	if (verbose) {
		std::string total_time = profiling.to_string();
		std::cerr << "ground total time: " << total_time << std::endl;
	}
	out_img.save(output_fname);
}

void heightmap_render(glm::vec3 *world_verts_cuda, int N, AABB* aabb,plane* ground_plane,  ppc &cur_ppc, glm::vec3* pixels, unsigned int* out_pixels, image &out_img, std::string output_fname) {
	int grid = 512, block = 32 * 4;
	timer profiling;
	profiling.tic();
	reset_pixel<<<grid, block>>> (vec3(0.0f), pixels);
	gpuErrchk(cudaDeviceSynchronize());
	
	raster_height << <grid, block>> > (world_verts_cuda, N, aabb, ground_plane, cur_ppc, pixels);
	
	gpuErrchk(cudaPeekAtLastError());	
	gpuErrchk(cudaDeviceSynchronize());

	to_unsigned_array << <grid, block >> > (w, h, 1, pixels, out_pixels);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	profiling.toc();
	if (verbose) {
		std::string total_time = profiling.to_string();
		std::cerr << "heightmap total time: " << total_time << std::endl;
	}
	out_img.save(output_fname);
}	

void render_data(const std::string model_file, const std::string output_folder) {
	std::shared_ptr<mesh> render_target;
    
	// load render target
	if (load_mesh(model_file, render_target)) {
		std::cerr << "Loading success \n";
	}
	else {
		std::cerr << "Loading failed \n";
	}
    
    cudaSetDevice(dev);
	// normalize and compute ground plane
	render_target->normalize_position_orientation();
	if (model_file.substr(model_file.size()-3) == "off") {
		render_target->m_world = glm::rotate(deg2rad(90.0f), vec3(-1.0f,0.0f,0.0f)) * render_target->m_world;
	}

	AABB world_aabb = render_target->compute_world_aabb();
	vec3 target_center = render_target->compute_world_center();

	vec3 lowest_point = world_aabb.p0;
	float offset = 0.0f - lowest_point.y;
	render_target->m_world = glm::translate(vec3(0.0, offset, 0.0)) * render_target->m_world;

	plane cur_plane = { vec3(0.0f), vec3(0.0f, 1.0f, 0.0f) };

	// set camera position
	std::shared_ptr<ppc> cur_ppc = std::make_shared<ppc>(w, h, 65.0f);
	float mesh_length = world_aabb.diag_length();
	if (mesh_length < 0.1f)
		mesh_length = 5.0f;
	vec3 new_pos = target_center + 2.0f * vec3(0.0f, mesh_length * 0.3f, mesh_length);
	cur_ppc->PositionAndOrient(new_pos, lowest_point, vec3(0.0f, 1.0f, 0.0f));

	// rasterize the 256x256 image to compute hard shadow
	purdue::create_folder(output_folder);
	mat4 old_mat = render_target->m_world;
	ppc  old_ppc = *cur_ppc;

	vec3 render_target_center = render_target->compute_world_center();
	float render_target_size = render_target->compute_world_aabb().diag_length();
	vec3 ppc_relative = vec3(0.0, 0.0, render_target_size) * 2.0f;
	std::vector<std::string> gt_str;
    
	timer t;
	t.tic();

	total_counter = cam_pitch.size() * model_rot.size() * 256 * 512 / patch_size / patch_size;
	std::cout << "total: " << total_counter << std::endl;
	int total_pixel = w * h;
	
	image out_img(w,h);
    glm::vec3* pixels, *tmp_pixels;
    unsigned int* out_pixels;
	plane* ground_plane;
	gpuErrchk(cudaMalloc((void**)&pixels, total_pixel * sizeof(glm::vec3)));
	gpuErrchk(cudaMalloc((void**)&tmp_pixels, total_pixel * sizeof(glm::vec3)));
	gpuErrchk(cudaMalloc(&out_pixels, total_pixel * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc(&ground_plane, sizeof(plane)));
	gpuErrchk(cudaMemcpy(ground_plane, &cur_plane, sizeof(plane), cudaMemcpyHostToDevice));

	int n = 100;
	std::vector<float> fov_states(n);
	for(int i = 0; i < n; ++i) {
		fov_states[i] = (float)i + 60.0f;
	}

	for(const auto& target_rot:model_rot) {
        render_target->m_world = glm::rotate(deg2rad(target_rot), glm::vec3(0.0, 1.0, 0.0)) * render_target->m_world;
		for (const auto& camera_pitch:cam_pitch) {
			// set camera rotation
			cur_ppc->PositionAndOrient(glm::rotate(deg2rad(camera_pitch), glm::vec3(-1.0, 0.0, 0.0)) * ppc_relative + render_target_center,
				render_target_center - vec3(0.0, render_target_center.y * 0.5f, 0.0),
				vec3(0.0, 1.0, 0.0));
			
			// random sample fovs
			float old_focal = cur_ppc->get_focal();
			float old_dist = glm::length(ppc_relative);

			// float random_fov = pd::normal_random(60,40);
			// while(random_fov<=20 || random_fov >=140) {
			// 	random_fov = pd::normal_random(60,40);
			// }
			// cur_ppc->set_fov(random_fov);
			
			for (auto new_fov:fov_states) {
				cur_ppc->set_fov(new_fov);

				float new_focal = cur_ppc->get_focal();
				float new_dist = old_dist / old_focal * new_focal;
				vec3 new_relative = vec3(0.0f,0.0f,new_dist);
				cur_ppc->PositionAndOrient(glm::rotate(deg2rad(camera_pitch), glm::vec3(-1.0, 0.0, 0.0)) * new_relative + render_target_center,
					render_target_center - vec3(0.0, render_target_center.y * 0.5f, 0.0),
					vec3(0.0, 1.0, 0.0));
				
				
				auto world_verts = render_target->compute_world_space_coords();
				AABB aabb = render_target->compute_world_aabb();
				glm::vec3* world_verts_cuda;
				AABB* aabb_cuda;
				gpuErrchk(cudaMalloc(&world_verts_cuda, world_verts.size() * sizeof(glm::vec3)));
				gpuErrchk(cudaMalloc(&aabb_cuda, sizeof(AABB)));
				gpuErrchk(cudaMemcpy(world_verts_cuda, world_verts.data(), world_verts.size() * sizeof(vec3), cudaMemcpyHostToDevice));
				gpuErrchk(cudaMemcpy(aabb_cuda, &aabb, sizeof(AABB), cudaMemcpyHostToDevice));

				// leave enough shadow space
				mask_render(world_verts_cuda, world_verts.size(), aabb_cuda, *cur_ppc, pixels, out_pixels, out_img, "");
				int highest_h = 0;
				for(int i = 0; i < w; ++i) {
					for(int j = 0; j < h; ++j) {
						if (out_img.at(i, j).x > 0.0f) {
							highest_h = std::max(j, highest_h);
						}
					}
				}
				// get rotation angle
				float focal = cur_ppc->get_focal();
				float ang_rad = pd::rad2deg(std::atan((float)(highest_h-256/2 + 2)/focal));
				cur_ppc->pitch(-(cur_ppc->get_fov() * 0.5f - ang_rad));
				
				std::string cur_prefix, output_fname;
				char buff[100]; snprintf(buff, sizeof(buff), "pitch_%d_rot_%d_fov_%d", (int)camera_pitch, (int)target_rot, (int)new_fov);
				cur_prefix = buff;

				// ------------------------------------ mask ------------------------------------ //
				if (render_mask) {
					output_fname = output_folder + "/" + cur_prefix + "_mask.png";
					mask_render(world_verts_cuda, world_verts.size(), aabb_cuda, *cur_ppc, pixels, out_pixels, out_img, output_fname);
				}

				// ------------------------------------ normal ------------------------------------ //
				if(render_normal) {
					output_fname = output_folder + "/" + cur_prefix + "_normal.png";
					normal_render(world_verts_cuda, world_verts.size(), aabb_cuda, *cur_ppc, pixels, out_pixels, out_img, output_fname);
				}

				// ------------------------------------ depth ------------------------------------ //
				if(render_depth) {
					output_fname = output_folder + "/" + cur_prefix + "_depth.png";
					depth_render(world_verts_cuda, world_verts.size(), aabb_cuda, *cur_ppc, pixels, out_pixels, out_img, output_fname);
				}

				// ------------------------------------ ground ------------------------------------ // 
				if (render_ground) {
					output_fname = output_folder + "/" + cur_prefix + "_ground.png";

					ground_render(ground_plane, *cur_ppc, pixels, out_pixels, out_img, output_fname);
				}
				
				// ------------------------------------ heightmap ------------------------------------ // 
				if (render_height) {
					output_fname = output_folder + "/" + cur_prefix + "_heightmap.png";

					heightmap_render(world_verts_cuda, world_verts.size(), aabb_cuda, ground_plane, *cur_ppc, pixels, out_pixels, out_img, output_fname);
				}

				// ------------------------------------ shadow ------------------------------------ // 
				if (render_shadow)
					shadow_render(world_verts_cuda, world_verts.size(), ground_plane, aabb_cuda, *cur_ppc, render_target_center,camera_pitch, target_rot, pixels, tmp_pixels, out_pixels, out_img);
				
				cudaFree(world_verts_cuda);
				cudaFree(aabb_cuda);
				*cur_ppc = old_ppc;
			}
		}
		
		// set back rotation
		render_target->m_world = old_mat;
	}
	cudaFree(pixels);
	cudaFree(tmp_pixels);
	cudaFree(out_pixels);
	cudaFree(ground_plane);

    t.toc();
	t.print_elapsed();
	std::string total_time = t.to_string();
	std::cout << "Finish this model: "<< total_time << "s"<< std::endl;
}

int main(int argc, char *argv[]) {
	try {
		cxxopts::Options options("shadow rendering", "Give me parameters for camera and model rotation");
		options
		.add_options()
		("cam_pitch", "A list of camera pitches", cxxopts::value<std::vector<float>>())
		("model_rot", "A list of model rotations", cxxopts::value<std::vector<float>>())
		("resume","Skip those rendered images?",cxxopts::value<bool>()->default_value("false"))
		("verbose","verbose time? ",cxxopts::value<bool>()->default_value("true"))
		("gpu","graphics card",cxxopts::value<int>()->default_value("0"))
		("model","model file path",cxxopts::value<std::string>())
		("output","output folder",cxxopts::value<std::string>())
		("render_shadow", "do you need to render shadow", cxxopts::value<bool>()->default_value("false"))
		("render_mask", "do you need to render mask", cxxopts::value<bool>()->default_value("false"))
		("render_normal", "do you need to render normal", cxxopts::value<bool>()->default_value("false"))
		("render_depth", "do you need to render depth", cxxopts::value<bool>()->default_value("false"))
		("render_ground", "do you need to render ground", cxxopts::value<bool>()->default_value("false"))
		("render_height", "do you need to render height", cxxopts::value<bool>()->default_value("false"))
		;
	
		auto result = options.parse(argc, argv);
	
		if (result.count("help"))
		{
		  std::cout << options.help() << std::endl;
		  exit(0);
		}
	
		if(result.count("cam_pitch")) {
			cam_pitch = result["cam_pitch"].as<std::vector<float>>();
			std::cout << "Camera pitch list: \n";
			for(const auto&v:cam_pitch) {
				printf(" %f \n", v);
			}
		} else {
			std::cout << "please set camera pitch \n";
			exit(0);
		}
	
		if(result.count("model_rot")) {
			model_rot = result["model_rot"].as<std::vector<float>>();
			std::cout << "Model rotation list: \n";
			
			for(const auto&v:model_rot) {
				printf("%f \n", v);
			}
		}else {
			std::cout << "please set model rotation \n";
			exit(0);
		}
	
		if(result.count("model")) {
			model_file = result["model"].as<std::string>();
		} else {
			std::cout << " please set model file path \n";
		}
	
		if(result.count("output")) {
			output_folder = result["output"].as<std::string>();
		} else {
			std::cout << " please set output folder \n";
		}
	
		dev = result["gpu"].as<int>();
		resume = result["resume"].as<bool>();
		verbose = result["verbose"].as<bool>();
		render_shadow = result["render_shadow"].as<bool>();
		render_mask = result["render_mask"].as<bool>();
		render_normal = result["render_normal"].as<bool>();
		render_depth = result["render_depth"].as<bool>();
		render_ground = result["render_ground"].as<bool>();
		render_height = result["render_height"].as<bool>();
		
		create_folder(output_folder);
	
		printf("begin render %s \n", model_file.c_str());
		printf("gridx: %d, gridy: %d, blockx: %d, blocky: %d", nx, ny, tx, ty);
		
		render_data(model_file, output_folder);
	
		printf("finished %s \n", model_file.c_str());
	}	
	catch (const cxxopts::OptionException& e) {
		std::cout << "error parsing options: " << e.what() << std::endl;
    	exit(1);
	}

	return 0;
}
