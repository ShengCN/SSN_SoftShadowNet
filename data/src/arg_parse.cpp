#include "arg_parse.h"

exp_params parse(int argc, char *argv[]) {
	exp_params ret;
	try {
		cxxopts::Options options("shadow rendering", "Give me parameters for camera and model rotation");
		options
		.add_options()
		("cam_pitch", "A list of camera pitches", cxxopts::value<std::vector<float>>())
		("model_rot", "A list of model rotations", cxxopts::value<std::vector<float>>())
		("resume","Skip those rendered images?",cxxopts::value<bool>()->default_value("true"))
		("verbose","verbose time? ",cxxopts::value<bool>()->default_value("true"))
		("gpu","graphics card",cxxopts::value<int>()->default_value("0"))
		("model_id","random seed",cxxopts::value<int>()->default_value("0"))
		("patch_size","patch size",cxxopts::value<int>()->default_value("8"))
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
		parse_by_key(result, "resume", ret.resume);
		parse_by_key(result, "verbose", ret.verbose);
		parse_by_key(result, "gpu", ret.gpu);
		parse_by_key(result, "patch_size", ret.patch_size);
		parse_by_key(result, "model_id", ret.model_id);
		parse_by_key(result, "model", ret.model);
		parse_by_key(result, "output", ret.output);
		parse_by_key(result, "render_shadow", ret.render_shadow);
		parse_by_key(result, "render_mask", ret.render_mask);
		parse_by_key(result, "render_normal", ret.render_normal);
		parse_by_key(result, "render_depth", ret.render_depth);
		parse_by_key(result, "render_touch", ret.render_touch);

		std::cout << (ret.to_string()) << std::endl;
	}	
	catch (const cxxopts::OptionException& e) {
		std::cout << "error parsing options: " << e.what() << std::endl;
    	exit(1);
	}
	return ret;
} 