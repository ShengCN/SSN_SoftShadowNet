#pragma once
#include <common.h>

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
		return pd::save_image(fname, pixels.data(), w, h);
	}
};

__host__ __device__
void set_pixel(vec3 c, unsigned int& p) {
	reinterpret_cast<unsigned char*>(&p)[0] = (unsigned char)(255.0 * c.x);
	reinterpret_cast<unsigned char*>(&p)[1] = (unsigned char)(255.0 * c.y);
	reinterpret_cast<unsigned char*>(&p)[2] = (unsigned char)(255.0 * c.z);
	reinterpret_cast<unsigned char*>(&p)[3] = (unsigned char)(255);
}

__global__
void reset_pixel(const vec3 c, glm::vec3* array, int w, int h) {
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
