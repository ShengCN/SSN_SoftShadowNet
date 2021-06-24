#pragma once
#include <common.h>

struct image {
	std::vector<glm::vec3> pixels;
	int w, h;

	image()=default; 

	image(int w, int h) :w(w), h(h) {
		clear();
	}

	void clear() {
		pixels.clear();
		pixels.resize(w * h, glm::vec3(0.0f));
	}

	void set_pixel(int u, int v, vec3 p) {
		size_t ind = (h - 1 - v) * w + u;
		pixels.at(ind) = p;
	}

	glm::vec3& at(int u, int v) {
		size_t ind = (h - 1 - v) * w + u;
		return pixels.at(ind);
	}

	glm::vec2 minmax() {
		glm::vec2 ret(FLT_MAX, -1e8);
		for(int i = 0; i < w * h; ++i) {
			ret.x = std::min(ret.x, pixels[i].x);
			ret.x = std::min(ret.x, pixels[i].y);
			ret.x = std::min(ret.x, pixels[i].z);

			ret.y = std::max(ret.y, pixels[i].x);
			ret.y = std::max(ret.y, pixels[i].y);
			ret.y = std::max(ret.y, pixels[i].z);
		}
		return ret;
	}

	bool save_image(const std::string fname);

	image operator+(const image &rhs);
	image operator*(const float &rhs);
};

__host__ __device__
void set_pixel(vec3 c, unsigned int& p);

__global__
void reset_pixel(const vec3 c, glm::vec3* array, int w, int h);

__global__
void to_unsigned_array(int w, int h, int patch_size, glm::vec3* array_a, unsigned int* array_out);

__global__
void add_array(int w, int h, glm::vec3* array_a, glm::vec3* array_out);
