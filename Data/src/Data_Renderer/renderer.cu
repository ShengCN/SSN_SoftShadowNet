#include "Data_Renderer.h"
#include "render_helper.h"
using namespace purdue;

/* Note different machine will generate different random numbers
 * Use precomputed random numbers insteaad
*/
std::vector<float> fov_list = { 50,44,42,42,47,49,48,44,40,54,48,56,57,51,47,53,59,50,47,48,58,57,43,54,42,44,51,43,47,49,45,51,53,52,56,51,43,43,56,51,42,52,47,42,45,52,46,50,56,48,57,51,54,56,55,53,59,50,44,59,46,48,45,54,56,42,56,   51,49,48,50,49,43,56,45,45,44,40,54,42,50,54,40,54,58,58,43,56,58,48,58,50,44,43,48,59,53,47,41,52};
/* Rendering Global Variables*/
float light_relative_length = 41.0f;    /* How far is the light */
int i_begin = 256/2;
timer profiling;

std::vector<pixel_pos> ibl_map;

// given an ibl map and light center position
// return the 3D light position
vec3 compute_light_pos(float x, float y, int w = 512, int h = 256) {
	//todo
	float x_fract = (float)x / w, y_fract = (float)y / h;
	purdue::deg alpha, beta;
	alpha = x_fract * 360.0f - 90.0f; beta = y_fract * 180.0f - 90.0f;
	return vec3(cos(deg2rad(beta)) * cos(deg2rad(alpha)), sin(deg2rad(beta)), cos(deg2rad(beta)) * sin(deg2rad(alpha)));
}

inline bool is_rendered(const std::string fname) {
	return exists_test(fname); 	
}

void mask_render(scene &cur_scene, render_param &cur_rp, output_param &out) {
	if (out.resume && is_rendered(out.ofname)) {
		std::cout << fmt::format("File {} skipped \n", out.ofname);
		return;
	}

	out.img = image(cur_rp.cur_ppc.width(), cur_rp.cur_ppc.height());
	cuda_container<glm::vec3> d_pixels(out.img.pixels);
	int block = 16 * 16, grid = (block + cur_rp.cur_ppc.width() * cur_rp.cur_ppc.height() - 1)/block;

	profiling.tic();
	raster_mask << <grid, block >> > (cur_scene.world_verts.get_d(), 
	cur_scene.world_verts.get_n(), 
	cur_scene.world_AABB.get_d(), 
	cur_rp.cur_ppc, 
	d_pixels.get_d());
	GC(cudaPeekAtLastError());
	GC(cudaDeviceSynchronize());
	d_pixels.mem_copy_back();
	profiling.toc();

	if (out.verbose) {
		std::string total_time = profiling.to_string();
		std::cout << "mask total time: " << total_time << std::endl;
	}

	if (out.ofname != "") {
		out.img.save_image(out.ofname);
	}
}

void rgb_render(scene &cur_scene, render_param &cur_rp, output_param &out) {
	if (out.resume && is_rendered(out.ofname)) {
		std::cout << fmt::format("File {} skipped \n", out.ofname);
		return;
	}

	out.img = image(cur_rp.cur_ppc.width(), cur_rp.cur_ppc.height());
	cuda_container<glm::vec3> d_pixels(out.img.pixels);
	int block = 16 * 16, grid = (block + cur_rp.cur_ppc.width() * cur_rp.cur_ppc.height() - 1)/block;

	profiling.tic();
	raster_rgb << <grid, block >> > (cur_scene.world_verts.get_d(), 
	cur_scene.world_verts.get_n(), 
	cur_scene.world_AABB.get_d(), 
	cur_rp.cur_ppc, 
	d_pixels.get_d());
	GC(cudaPeekAtLastError());
	GC(cudaDeviceSynchronize());
	d_pixels.mem_copy_back();
	profiling.toc();

	if (out.verbose) {
		std::string total_time = profiling.to_string();
		std::cout << "rgb total time: " << total_time << std::endl;
	}

	if (out.ofname != "") {
		out.img.save_image(out.ofname);
	}
}

void normal_render(scene &cur_scene, render_param &cur_rp, output_param &out) {
	if (out.resume && is_rendered(out.ofname)) {
		std::cout << fmt::format("File {} skipped \n", out.ofname);
		return;
	}

	out.img = image(cur_rp.cur_ppc.width(), cur_rp.cur_ppc.height());
	cuda_container<glm::vec3> d_pixels(out.img.pixels);
	int block = 16 * 16, grid = (block + cur_rp.cur_ppc.width() * cur_rp.cur_ppc.height() - 1)/block;

	profiling.tic();
	raster_normal << <grid, block >> > (cur_scene.world_verts.get_d(), 
	cur_scene.world_verts.get_n(), 
	cur_scene.world_AABB.get_d(), 
	cur_rp.cur_ppc, 
	d_pixels.get_d());
	GC(cudaPeekAtLastError());
	GC(cudaDeviceSynchronize());
	profiling.toc();
	d_pixels.mem_copy_back();

	if (out.verbose) {
		std::string total_time = profiling.to_string();
		std::cout << "Normal total time: " << total_time << std::endl;
	}

	if (out.ofname != "") {
		out.img.save_image(out.ofname);
	}
}

void depth_render(scene &cur_scene, render_param &cur_rp, output_param &out) {
	if (out.resume && is_rendered(out.ofname)) {
		std::cout << fmt::format("File {} skipped \n", out.ofname);
		return;
	}

	out.img = image(cur_rp.cur_ppc.width(), cur_rp.cur_ppc.height());
	cuda_container<glm::vec3> d_pixels(out.img.pixels);
	int block = 16 * 16, grid = (block + cur_rp.cur_ppc.width() * cur_rp.cur_ppc.height() - 1)/block;

	profiling.tic();
	raster_depth << <grid, block >> > (cur_scene.world_verts.get_d(), 
	cur_scene.world_verts.get_n(), 
	cur_scene.world_AABB.get_d(), 
	cur_rp.cur_ppc, 
	d_pixels.get_d());
	GC(cudaPeekAtLastError());
	GC(cudaDeviceSynchronize());
	profiling.toc();
	d_pixels.mem_copy_back();

	if (out.verbose) {
		std::string total_time = profiling.to_string();
		std::cout << "Depth total time: " << total_time << std::endl;
	}

	if (out.ofname != "") {
		out.img.save_image(out.ofname);
	}
}

void touch_render(scene &cur_scene, render_param &cur_rp, output_param &out) {
	if (out.resume && is_rendered(out.ofname)) {
		std::cout << fmt::format("File {} skipped \n", out.ofname);
		return;
	}

	out.img = image(cur_rp.cur_ppc.width(), cur_rp.cur_ppc.height());
	cuda_container<glm::vec3> d_pixels(out.img.pixels);
	int block = 16 * 16, grid = (block + cur_rp.cur_ppc.width() * cur_rp.cur_ppc.height() - 1)/block;

	profiling.tic();
	raster_touch << <grid, block >> > (cur_scene.world_verts.get_d(), 
	cur_scene.world_verts.get_n(), 
	cur_scene.world_AABB.get_d(), 
	cur_scene.ground_plane.get_d(),
	cur_rp.cur_ppc, 
	cur_scene.rd_samples.get_d(),
	cur_scene.rd_samples.get_n(),
	d_pixels.get_d());
	GC(cudaPeekAtLastError());
	GC(cudaDeviceSynchronize());
	profiling.toc();
	d_pixels.mem_copy_back();

	if (out.verbose) {
		std::string total_time = profiling.to_string();
		std::cout << "Touch total time: " << total_time << std::endl;
	}

	if (out.ofname != "") {
		out.img.save_image(out.ofname);
	}
}

void height_render(scene &cur_scene, render_param &cur_rp, output_param &out) {
	if (out.resume && is_rendered(out.ofname)) {
		std::cout << fmt::format("File {} skipped \n", out.ofname);
		return;
	}

	out.img = image(cur_rp.cur_ppc.width(), cur_rp.cur_ppc.height());
	cuda_container<glm::vec3> d_pixels(out.img.pixels);
	int block = 16 * 16, grid = (block + cur_rp.cur_ppc.width() * cur_rp.cur_ppc.height() - 1)/block;

	profiling.tic();
	raster_height << <grid, block >> > (cur_scene.world_verts.get_d(), 
	cur_scene.world_verts.get_n(), 
	cur_scene.world_AABB.get_d(), 
	cur_scene.ground_plane.get_d(),
	cur_rp.cur_ppc, 
	d_pixels.get_d());
	GC(cudaPeekAtLastError());
	GC(cudaDeviceSynchronize());
	profiling.toc();
	d_pixels.mem_copy_back();

	if (out.verbose) {
		std::string total_time = profiling.to_string();
		std::cout << "Height total time: " << total_time << std::endl;
	}

	if (out.ofname != "") {
		out.img.save_image(out.ofname);
	}
}

void shadow_render(scene &cur_scene, render_param &cur_rp, output_param &out) {
	out.img = image(cur_rp.cur_ppc.width(), cur_rp.cur_ppc.height());
	image tmp_img(cur_rp.cur_ppc.width(), cur_rp.cur_ppc.height());
	
	int counter = 0, total_counter =cur_rp.ibl_w/cur_rp.patch_size *cur_rp.ibl_h / cur_rp.patch_size/2;  
	for(int ibl_i = 0; ibl_i <= cur_rp.ibl_w/cur_rp.patch_size; ++ibl_i) {
		for(int ibl_j = cur_rp.ibl_h / 2 /cur_rp.patch_size; ibl_j <= cur_rp.ibl_h / cur_rp.patch_size; ++ibl_j) {
			pixel_pos begin_pixel_pos = {ibl_i * cur_rp.patch_size, ibl_j * cur_rp.patch_size};
			std::string cur_prefix = fmt::format("pitch_{:.5f}_rot_{:.5f}_fov_{:.2f}_ibli_{}_iblj_{}", cur_rp.camera_pitch, cur_rp.target_rot, cur_rp.cur_ppc.get_fov(), begin_pixel_pos.x, begin_pixel_pos.y);
			std::string output_fname = out.output_folder + "/" + cur_prefix + "_shadow.png";
			
			if (out.resume && is_rendered(output_fname)) {
				std::cout << fmt::format("File {} skipped \n", output_fname);
				counter++; 
				continue;
			}
			
			int block = 16 * 16, grid = (block + cur_rp.cur_ppc.width() * cur_rp.cur_ppc.height() - 1)/block;
			out.img.clear();

			profiling.tic();
			for(int patch_i = 0; patch_i < cur_rp.patch_size; ++patch_i)
				for(int patch_j = 0; patch_j < cur_rp.patch_size; ++patch_j) {
					/* If not average patch, skip non-center samples */
					if (!out.base_avg) {
						if ((patch_i != cur_rp.patch_size/2 || patch_j != cur_rp.patch_size/2)) {
							continue;
						} 					
					}

					tmp_img.clear();
					cuda_container<glm::vec3> dtmp_img(tmp_img.pixels);

					for (int si = 0; si < cur_rp.base_samples; ++si) {
						pixel_pos light_pixel_pos = {begin_pixel_pos.x + patch_i + pd::random_float(), begin_pixel_pos.y + patch_j + pd::random_float()};

						cur_rp.light_pos = compute_light_pos(light_pixel_pos.x, light_pixel_pos.y, cur_rp.ibl_w, cur_rp.ibl_h) * 400000.0f + cur_rp.render_target_center;

						raster_hard_shadow<<<grid, block>>>(
							cur_scene.ground_plane.get_d(),
							cur_scene.world_verts.get_d(),
							cur_scene.world_verts.get_n(),
							cur_scene.world_AABB.get_d(),
							cur_rp.cur_ppc,
							cur_rp.light_pos,
							dtmp_img.get_d());
						GC(cudaPeekAtLastError());
						GC(cudaDeviceSynchronize());

						dtmp_img.mem_copy_back();

						float weight = 1.0f;
						if (out.base_avg) {
							weight = 1.0f/(cur_rp.patch_size * cur_rp.patch_size * cur_rp.base_samples);
						}
						out.img = out.img + tmp_img * weight;
					}
			}
			profiling.toc();

			if (out.verbose) {
				std::cout << fmt::format("{},{} shadow render: {}s \n", 
				begin_pixel_pos.x, begin_pixel_pos.y, profiling.to_string());
			}

			out.img.save_image(output_fname);
			std::cerr << "Finish: " << (float)++counter / total_counter * 100.0f << "% \r";
		}
	}
}

/* Initialize the scene
 *  1. load mesh
 * 	1. normalize the mesh size
 *  2. set up camera 
*/
void scene_initialize(const exp_params &params, 
	std::shared_ptr<mesh> &render_target) {
	// load render target
	if (load_mesh(params.model, render_target)) {
		std::cout << fmt::format("{} Loading success, Triangle: {} \n",params.model, render_target->m_verts.size());
	}
	else {
		std::cerr << params.model + " Loading failed \n";
		exit(0);
	}

	// normalize and compute ground plane
	render_target->normalize_position_orientation();
	if (params.model.substr(params.model.size()-3) == "off") {
		render_target->m_world = glm::rotate(deg2rad(90.0f), vec3(-1.0f,0.0f,0.0f)) * render_target->m_world;
	}

	AABB world_aabb = render_target->compute_world_aabb();
	vec3 lowest_point = world_aabb.p0;
	float offset = 0.0f - lowest_point.y;
	render_target->m_world = glm::translate(vec3(0.0, offset, 0.0)) * render_target->m_world;
} 

void render_scenes(const exp_params &params, 
	std::shared_ptr<mesh> &render_target, 
	std::shared_ptr<ppc> &render_camera) {
	plane cur_plane = { vec3(0.0f,0.0f,0.0f), vec3(0.0f, 1.0f, 0.0f) };
	vec3 render_target_center = render_target->compute_world_center();
	vec3 ppc_relative = vec3(0.0, 0.0, render_target->compute_world_aabb().diag_length()) * 2.0f;
	std::vector<vec3> samples = uniform_sphere_3d_samples(1000);

	mat4 old_mat = render_target->m_world;
	ppc  old_ppc = *render_camera;
	
	/* Iterations for one model */
	int total_counter = params.cam_pitch.size() * 
	params.model_rot.size() * 
	params.ibl_w * params.ibl_h / 
	(params.patch_size * params.patch_size);
	std::cout << "total: " << total_counter << std::endl;

	for(const auto& target_rot:params.model_rot) {
		/* New object orientation */
        render_target->m_world = glm::rotate(deg2rad(target_rot), glm::vec3(0.0, 1.0, 0.0)) * render_target->m_world;
		for (const auto& camera_pitch:params.cam_pitch) {
			/* New camera position */
			render_camera->PositionAndOrient(
				glm::rotate(deg2rad(camera_pitch), glm::vec3(-1.0, 0.0, 0.0)) * ppc_relative + render_target_center,
				render_target_center - vec3(0.0, render_target_center.y * 0.5f, 0.0),
				vec3(0.0, 1.0, 0.0));
			
			/* Random Fov */
            float random_fov = (float)fov_list[params.model_id % (int)fov_list.size()];
			render_camera->set_fov(random_fov);
			render_camera->PositionAndOrient(glm::rotate(deg2rad(camera_pitch), glm::vec3(-1.0, 0.0, 0.0)) * ppc_relative + render_target_center,
				render_target_center - vec3(0.0, render_target_center.y * 0.5f, 0.0),
				vec3(0.0, 1.0, 0.0));
			
			auto world_verts = render_target->compute_world_space_coords();
			AABB aabb = render_target->compute_world_aabb();

			/* Set up scene and rendering parameters */
			scene cur_scene(world_verts, 
				aabb, 
				samples, 
				render_target->compute_world_center(),
				cur_plane);
			render_param cur_rp = {*render_camera, vec3(0.0f), render_target_center, 
				params.ibl_h, params.ibl_w, params.patch_size, params.base_samples, camera_pitch, target_rot};
			output_param oparam = {params.resume, params.verbose, params.base_avg, params.output, "", image()};

			/* Refine the camera orientation to leave enough shadow space */
			mask_render(cur_scene, cur_rp, oparam);
			int highest_h = 0;
			for(int i = 0; i < params.w; ++i) for(int j = 0; j < params.h; ++j) {
				if (oparam.img.at(i, j).x > 0.0f) {
					highest_h = std::max(j, highest_h);
				}
			}

			// get rotation angle
			float ang_rad = pd::rad2deg(std::atan((float)(highest_h - params.h/2 + 2)/render_camera->get_focal()));
            std::cout << "pitch angle: " << std::atan((float)(highest_h - params.h/2 + 2)/render_camera->get_focal()) << std::endl;
			render_camera->pitch(-(render_camera->get_fov() * 0.5f - ang_rad));
			std::cout << fmt::format("Highest: {} Angle: {}, pos: {} ori: {}\n", highest_h, ang_rad, to_string(render_camera->_position), to_string(render_camera->_front));
			cur_rp.cur_ppc = *render_camera;
			
			std::string cur_prefix;
			cur_prefix = fmt::format("pitch_{:.5f}_rot_{:.5f}_fov_{:.2f}", camera_pitch, target_rot, random_fov);

            {
				oparam.ofname = params.output + "/" + cur_prefix + "_mts.bin";
                std::fstream oss(oparam.ofname, std::ios::binary | std::ios::out);

                oss.write((char*)&cur_rp.cur_ppc, sizeof(ppc));
                auto mat = render_target->get_world_mat();
                oss.write((char*)&mat, sizeof(mat));
                auto fpath = render_target->file_path;
                size_t flen = fpath.size();
                oss.write((char*)&flen, sizeof(size_t));
                oss.write((char*)&fpath[0], flen);
                oss.close();
            }

			// ------------------------------------ mask ------------------------------------ //
			if (params.render_mask) {
				oparam.ofname = params.output + "/" + cur_prefix + "_mask.png";
				mask_render(cur_scene, cur_rp, oparam);
			}

			// ------------------------------------ rgb ------------------------------------ //
			if (params.render_mask) {
				oparam.ofname = params.output + "/" + cur_prefix + "_rgb.png";
				rgb_render(cur_scene, cur_rp, oparam);
			}

			// ------------------------------------ normal ------------------------------------ //
			if(params.render_normal) {
				oparam.ofname = params.output + "/" + cur_prefix + "_normal.png";
				normal_render(cur_scene, cur_rp, oparam);
			}

			// ------------------------------------ depth ------------------------------------ //
			if(params.render_depth) {
				oparam.ofname = params.output + "/" + cur_prefix + "_depth.png";
				depth_render(cur_scene, cur_rp, oparam);
			}
			
			// ------------------------------------ touching ------------------------------------ // 
			if (params.render_touch) {
				oparam.ofname = params.output + "/" + cur_prefix + "_touch.png";
				touch_render(cur_scene, cur_rp, oparam);
			}

			// ------------------------------------ height ------------------------------------ // 
			if (params.render_height) {
				oparam.ofname = params.output + "/" + cur_prefix + "_height.png";
				height_render(cur_scene, cur_rp, oparam);
			}

			// ------------------------------------ shadow ------------------------------------ // 
			if (params.render_shadow) {
				shadow_render(cur_scene, cur_rp, oparam);
			}
			
			*render_camera = old_ppc;
		}
		
		// set back rotation
		render_target->m_world = old_mat;
	}
}

void render_data(const exp_params &params) {
	timer t; 
	std::shared_ptr<mesh> render_target = std::make_shared<mesh>();
	std::shared_ptr<ppc> cur_ppc = std::make_shared<ppc>(params.w, params.h, 50.0f);
    
	scene_initialize(params, render_target);

    cudaSetDevice(params.gpu);
	purdue::create_folder(params.output);
 
	t.tic();
	render_scenes(params, render_target, cur_ppc);
    t.toc(); 
	std::cout << fmt::format("Finish {}: {}s \n", params.model, t.to_string()); 
}
