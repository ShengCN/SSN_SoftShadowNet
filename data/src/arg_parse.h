#pragma once
#include <common.h>
#include "cxxopts.h"

struct exp_params {
	std::vector<float> cam_pitch, model_rot;
	bool resume, verbose, render_shadow, render_mask, render_normal, render_depth, render_touch;
	int gpu, model_id;
	std::string model, output;

	exp_params()=default; 

	std::string to_string() {
		std::stringstream oss;
		oss << "--------------------------------------------------" << std::endl;
		oss << fmt::format("cam_pitch: {}".format(cam_pitch)) << std::endl; 
		oss << fmt::format("model_rot: {}".format(model_rot)) << std::endl; 
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
	} else {
		ERROR(fmt::format("Please give {} input", key));
		exit(0);
	}
}

exp_params parse(int argc, char *argv[]) {
	exp_params ret;
	try {
		cxxopts::Options options("shadow rendering", "Give me parameters for camera and model rotation");
		options
		.add_options()
		("cam_pitch", "A list of camera pitches", cxxopts::value<std::vector<float>>())
		("model_rot", "A list of model rotations", cxxopts::value<std::vector<float>>())
		("resume","Skip those rendered images?",cxxopts::value<bool>()->default_value("false"))
		("verbose","verbose time? ",cxxopts::value<bool>()->default_value("true"))
		("gpu","graphics card",cxxopts::value<int>()->default_value("0"))
		("model_id","random seed",cxxopts::value<int>()->default_value("0"))
		("model","model file path",cxxopts::value<std::string>())
		("output","output folder",cxxopts::value<std::string>())
		("render_shadow", "do you need to render shadow", cxxopts::value<bool>()->default_value("false"))
		("render_mask", "do you need to render mask", cxxopts::value<bool>()->default_value("false"))
		("render_normal", "do you need to render normal", cxxopts::value<bool>()->default_value("false"))
		("render_depth", "do you need to render depth", cxxopts::value<bool>()->default_value("false"))
		("render_touch", "do you need to render touch", cxxopts::value<bool>()->default_value("false"))
		;
	
		auto result = options.parse(argc, argv);
		if (result.count("help")) {
		  std::cout << options.help() << std::endl;
		  exit(0);
		}
	
		parse_by_key(result, "cam_pitch", ret.cam_pitch);
		parse_by_key(result, "model_rot", ret.model_rot);
		parse_by_key(result, "out_prefix", ret.resume);
		parse_by_key(result, "lens_effect", ret.verbose);
		parse_by_key(result, "lens_size", ret.gpu);
		parse_by_key(result, "focal_plane", ret.model_id);
		parse_by_key(result, "mc_sample_num", ret.model);
		parse_by_key(result, "norm", ret.output);
		parse_by_key(result, "algo", ret.render_shadow);
		parse_by_key(result, "out_folder", ret.render_mask);
		parse_by_key(result, "gpu", ret.render_normal);
		parse_by_key(result, "gpu", ret.render_depth);
		parse_by_key(result, "gpu", ret.render_touch);

		INFO(ret.to_string());
	}	
	catch (const cxxopts::OptionException& e) {
		std::cout << "error parsing options: " << e.what() << std::endl;
    	exit(1);
	}
	return ret;
} 