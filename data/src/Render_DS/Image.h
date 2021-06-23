#pragma once
#include <common.h>

struct image {
	std::vector<unsigned int> pixels;
	int w, h;
	image(int w, int h) :w(w), h(h) {
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

	bool save_image(const std::string fname);
};

__host__ __device__
void set_pixel(vec3 c, unsigned int& p);

__global__
void reset_pixel(const vec3 c, glm::vec3* array, int w, int h);

__global__
void to_unsigned_array(int w, int h, int patch_size, glm::vec3* array_a, unsigned int* array_out);

__global__
void add_array(int w, int h, glm::vec3* array_a, glm::vec3* array_out);
