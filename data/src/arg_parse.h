#pragma once
#include <common.h>

struct exp_params {
	std::vector<float> cam_pitch, model_rot;
	bool resume, verbose, render_shadow, render_mask, render_normal, render_depth, render_touch;
	int gpu, model_id, w, h, ibl_w, ibl_h, patch_size;
	std::string model, output;

	exp_params() {
		/* Default values here */
		resume = true;
		verbose = true;
		render_shadow = render_mask = render_normal = render_depth = render_touch = false; 
		gpu = 0;
		w = h = 256;
		ibl_w = 512;
		ibl_h = 128;
		patch_size = 8;

		/* Used for indexing some random numbers */
		model_id = 0;
	} 

	std::string to_string() {
		std::stringstream oss;
		oss << "--------------------------------------------------" << std::endl;
		oss << fmt::format("cam_pitch: {}", cam_pitch) << std::endl; 
		oss << fmt::format("model_rot: {}", model_rot) << std::endl; 
		oss << "resume " << resume << std::endl;
		oss << "verbose " << verbose << std::endl;
		oss << "render_shadow " << render_shadow << std::endl;
		oss << "render_mask " << render_mask << std::endl;
		oss << "render_normal " << render_normal << std::endl;
		oss << "render_depth " << render_depth << std::endl;
		oss << "render_touch " << render_touch << std::endl;
		oss << "gpu: " << gpu << std::endl;
		oss << "model_id(for data consistent): " << model_id << std::endl;
		oss << "model: " << model << std::endl;
		oss << "output: " << output << std::endl;
		oss << "--------------------------------------------------" << std::endl;
		return oss.str();
	}
};

template<typename T> 
void parse_by_key(const cxxopts::ParseResult &result, std::string key, T &out) {
	if (result.count(key)) {
		out = result[key].as<T>();
	} 
}

exp_params parse(int argc, char *argv[]);