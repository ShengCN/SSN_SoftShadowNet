
#include "Image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

__host__ __device__
glm::vec3 unsign_vec3(unsigned int v) {
	glm::vec3 ret(0.0f);
	ret.r = reinterpret_cast<unsigned char*>(&(ret))[0]/255.0f;
	ret.g = reinterpret_cast<unsigned char*>(&(ret))[1]/255.0f;
	ret.b = reinterpret_cast<unsigned char*>(&(ret))[2]/255.0f;
	return ret;
}

__host__ __device__
unsigned int vec3_unsign(glm::vec3 v) {
	unsigned int ret=0xff000000;
	reinterpret_cast<unsigned char*>(&(ret))[0] = (unsigned char)(255.0 * v.r);
	reinterpret_cast<unsigned char*>(&(ret))[1] = (unsigned char)(255.0 * v.g);
	reinterpret_cast<unsigned char*>(&(ret))[2] = (unsigned char)(255.0 * v.b);
	return ret;
}

bool image::save_image(const std::string fname){
	std::vector<unsigned int> buffer(w * h);
	for(int i = 0; i < w * h; ++i) {
		buffer[i] = vec3_unsign(glm::clamp(pixels[i], 0.0f, 1.0f));
	}

	return stbi_write_png(fname.c_str(), w, h, 4, buffer.data(), w*4);
}

image image::operator+(const image &rhs) {
	image ret(w, h);

	for(int i = 0; i < w * h; ++i) {
		ret.pixels[i] = pixels[i] + rhs.pixels[i];
	}
	return ret;
}

image image::operator*(const float &rhs) {
	image ret = *this;

	for(int i = 0; i < w * h; ++i) {
		ret.pixels[i] = pixels[i] * rhs;
	}
	return ret;
}

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
