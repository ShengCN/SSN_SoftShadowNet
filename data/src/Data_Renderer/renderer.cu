#include "Data_Renderer.h"
#include "render_helper.h"

void mask_render(scene &cur_scene, render_param &cur_rp, image &out_img) {
	
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
	for(int ibl_i = 0; ibl_i <= 512/params.patch_size; ++ibl_i) {
		for(int ibl_j = i_begin/params.patch_size; ibl_j <= 256/params.patch_size; ++ibl_j) {
			pixel_pos begin_pixel_pos = {ibl_i * params.patch_size, ibl_j * params.patch_size};
			// vec3 light_position = compute_light_pos(begin_pixel_pos.x, begin_pixel_pos.y) * light_relative_length + render_target_center;
			// vec3 light_position = compute_light_pos(begin_pixel_pos.x, begin_pixel_pos.y) * 400.0f + render_target_center;
			
			std::string cur_prefix;
			float fov = cur_ppc.get_fov();
			char buff[100]; snprintf(buff, sizeof(buff), "pitch_%d_rot_%d_fov_%d_ibli_%d_iblj_%d", (int)camera_pitch, (int)target_rot, (int)fov,begin_pixel_pos.x, begin_pixel_pos.y);
			cur_prefix = buff;
			std::string output_fname = params.output + "/" + cur_prefix + "_shadow.png";
			
			if(params.resume) {
				if(exists_test(output_fname)) {
					std::cout << output_fname << " is skipped \n";
					++counter;
					continue;
				}
			}
			
			profiling.tic();
			int block = 16 * 16, grid = (block + params.w * params.h - 1)/block;
			reset_pixel<<<grid, block>>>(vec3(0.0f), pixels, cur_ppc._width, cur_ppc._height);
			gpuErrchk(cudaDeviceSynchronize());
			for(int patch_i = 0; patch_i < params.patch_size; ++patch_i)
				for(int patch_j = 0; patch_j < params.patch_size; ++patch_j) {
					pixel_pos light_pixel_pos = {begin_pixel_pos.x + patch_i, begin_pixel_pos.y + patch_j};
					vec3 light_position = compute_light_pos(light_pixel_pos.x, light_pixel_pos.y) * 400000.0f + render_target_center;
					
					reset_pixel<<<grid, block>>>(vec3(0.0f), tmp_pixels, cur_ppc._width, cur_ppc._height);
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
					
					add_array<<<grid, block>>>(params.w, params.h, tmp_pixels, pixels);
					gpuErrchk(cudaDeviceSynchronize());
			}
			to_unsigned_array<<<grid, block>>>(params.w, params.h, params.patch_size, pixels, out_pixels);
			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
			
			profiling.toc();
			if (params.verbose) {
				std::string total_time = profiling.to_string();
				std::cerr << begin_pixel_pos.x << "," << begin_pixel_pos.y << " shadow rendering time: " << total_time << std::endl;
			}

			out_img.save(output_fname);
			std::cerr << "Finish: " << (float)++counter / total_counter * 100.0f << "% \r";
		}
	}
	// ----------------------------------------------- SHADOW BASES ----------------------------------------------------
}

void mask_render(glm::vec3* world_verts_cuda, int N, AABB* aabb_cuda, ppc &cur_ppc, glm::vec3* pixels, unsigned int* out_pixels, image &out_img, std::string output_fname) {
	// ----------------------------------------------- Masks ----------------------------------------------------
	if(params.resume) {
		if(exists_test(output_fname)) {
			std::cout << output_fname << " is skipped \n";
			return;
		}
	}
	int block = 16 * 16, grid = (block + params.w * params.h - 1)/block;

	timer profiling;
	profiling.tic();
	reset_pixel <<<grid, block >> > (vec3(0.0f), pixels, cur_ppc._width, cur_ppc._height);
	gpuErrchk(cudaDeviceSynchronize());
	raster_mask << <grid, block >> > (world_verts_cuda,
		N,
		aabb_cuda,
		cur_ppc,
		pixels);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	to_unsigned_array << <grid, block >> > (params.w, params.h, 1, pixels, out_pixels);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	profiling.toc();
	if (params.verbose) {
		std::string total_time = profiling.to_string();
		std::cerr << "mask total time: " << total_time << std::endl;
	}

	if (output_fname != "")
		out_img.save(output_fname);
	// ----------------------------------------------- Masks ----------------------------------------------------
}

void normal_render(glm::vec3* world_verts_cuda, int N, AABB* aabb_cuda, ppc &cur_ppc, glm::vec3* pixels, unsigned int* out_pixels, image &out_img, std::string output_fname) {
	// ----------------------------------------------- Normal ----------------------------------------------------
	if(params.resume) {
		if(exists_test(output_fname)) {
			std::cout << output_fname << " is skipped \n";
			return;
		}
	}
	int block = 16 * 16, grid = (block + params.w * params.h - 1)/block;

	timer profiling;
	profiling.tic();
	reset_pixel << <grid, block >> > (vec3(0.0f), pixels, cur_ppc._width, cur_ppc._height);
	gpuErrchk(cudaDeviceSynchronize());
	raster_normal << <grid, block >> > (world_verts_cuda,
		N,
		aabb_cuda,
		cur_ppc,
		pixels);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	to_unsigned_array << <grid, block >> > (params.w, params.h, 1, pixels, out_pixels);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	profiling.toc();
	if (params.verbose) {
		std::string total_time = profiling.to_string();
		std::cerr << "normal total time: " << total_time << std::endl;
	}
	out_img.save(output_fname);
	// ----------------------------------------------- Normal ----------------------------------------------------
}


void depth_render(glm::vec3* world_verts_cuda, int N, AABB* aabb_cuda, ppc &cur_ppc, glm::vec3* pixels, unsigned int* out_pixels, image &out_img, std::string output_fname) {
	// ----------------------------------------------- Depth ----------------------------------------------------
	if(params.resume) {
		if(exists_test(output_fname)) {
			std::cout << output_fname << " is skipped \n";
			return;
		}
	}
	int block = 16 * 16, grid = (block + params.w * params.h - 1)/block;

	timer profiling;
	profiling.tic();
	reset_pixel << <grid, block >> > (vec3(0.0f), pixels, cur_ppc._width, cur_ppc._height);
	gpuErrchk(cudaDeviceSynchronize());
	raster_depth << <grid, block >> > (world_verts_cuda,
		N,
		aabb_cuda,
		cur_ppc,
		pixels);
	gpuErrchk(cudaPeekAtLastError());	
	gpuErrchk(cudaDeviceSynchronize());

	to_unsigned_array << <grid, block >> > (params.w, params.h, 1, pixels, out_pixels);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	profiling.toc();
	if (params.verbose) {
		std::string total_time = profiling.to_string();
		std::cerr << "depth total time: " << total_time << std::endl;
	}
	out_img.save(output_fname);
	// ----------------------------------------------- Depth ----------------------------------------------------
}

void ground_render(plane* ground_plane,  ppc &cur_ppc, glm::vec3* pixels, unsigned int* out_pixels, image &out_img, std::string output_fname) {
	if(params.resume) {
		if(exists_test(output_fname)) {
			std::cout << output_fname << " is skipped \n";
			return;
		}
	}
	int block = 16 * 16, grid = (block + params.w * params.h - 1)/block;
	timer profiling;
	profiling.tic();
	reset_pixel<<<grid, block>>> (vec3(0.0f), pixels, cur_ppc._width, cur_ppc._height);
	gpuErrchk(cudaDeviceSynchronize());
	
	raster_ground << <grid, block>> > (ground_plane, cur_ppc, pixels);
	
	gpuErrchk(cudaPeekAtLastError());	
	gpuErrchk(cudaDeviceSynchronize());

	to_unsigned_array << <grid, block >> > (params.w, params.h, 1, pixels, out_pixels);
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	profiling.toc();
	if (params.verbose) {
		std::string total_time = profiling.to_string();
		std::cerr << "ground total time: " << total_time << std::endl;
	}
	out_img.save(output_fname);
}

void touch_render(glm::vec3 *world_verts_cuda, int N, AABB* aabb,plane* ground_plane,  ppc &cur_ppc, glm::vec3* pixels, unsigned int* out_pixels, image &out_img, std::string output_fname) {
	std::cout << "Begin touch: " << output_fname << std::endl;
	if(params.resume) {
		if(exists_test(output_fname)) {
			std::cout << output_fname << " is skipped \n";
			return;
		}
	}
	int block = 16 * 16, grid = (block + params.w * params.h - 1)/block;
	timer profiling;
	profiling.tic();
	reset_pixel<<<grid, block>>> (vec3(0.0f), pixels, cur_ppc._width, cur_ppc._height);
	gpuErrchk(cudaDeviceSynchronize());
	
	auto samples = uniform_sphere_3d_samples(1000);
	vec3 *cuda_samples;
	cudaMalloc(&cuda_samples, sizeof(vec3) * samples.size());
	cudaMemcpy(cuda_samples, samples.data(), (int)samples.size(), cudaMemcpyHostToDevice);

	raster_touch << <grid, block>> > (world_verts_cuda, N, aabb, ground_plane, cur_ppc, cuda_samples, (int)samples.size(), pixels);
	
	GC(cudaPeekAtLastError());	
	gpuErrchk(cudaDeviceSynchronize());

	to_unsigned_array << <grid, block >> > (params.w, params.h, 1, pixels, out_pixels);
	GC(cudaDeviceSynchronize());
	GC(cudaMemcpy((unsigned int*)&out_img.pixels[0], out_pixels, out_img.pixels.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	profiling.toc();
	if (params.verbose) {
		std::string total_time = profiling.to_string();
		std::cerr << "touch total time: " << total_time << std::endl;
	}
	out_img.save(output_fname);
	cudaFree(cuda_samples);
}	

void render_data(const std::string model_file, const std::string output_folder) {
	std::shared_ptr<mesh> render_target;
    
	// load render target
	if (load_mesh(model_file, render_target)) {
		std::cerr << model_file + " Loading success \n";
	}
	else {
		std::cerr << model_file + " Loading failed \n";
	}
    
    cudaSetDevice(params.gpu);
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

	plane cur_plane = { vec3(0.0f,0.0f,0.0f), vec3(0.0f, 1.0f, 0.0f) };

	// set camera position
	std::shared_ptr<ppc> cur_ppc = std::make_shared<ppc>(params.w, params.h, 65.0f);
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

	total_counter = params.cam_pitch.size() * params.model_rot.size() * 256 * 512 / params.patch_size / params.patch_size;
	std::cout << "total: " << total_counter << std::endl;
	int total_pixel = params.w * params.h;
	
	image out_img(params.w,params.h);
    glm::vec3* pixels, *tmp_pixels;
    unsigned int* out_pixels;
	plane* ground_plane;
	GC(cudaMalloc((void**)&pixels, total_pixel * sizeof(glm::vec3)));
	GC(cudaMalloc((void**)&tmp_pixels, total_pixel * sizeof(glm::vec3)));
	GC(cudaMalloc(&out_pixels, total_pixel * sizeof(unsigned int)));
	GC(cudaMalloc(&ground_plane, sizeof(plane)));
	GC(cudaMemcpy(ground_plane, &cur_plane, sizeof(plane), cudaMemcpyHostToDevice));

	for(const auto& target_rot:params.model_rot) {
        render_target->m_world = glm::rotate(deg2rad(target_rot), glm::vec3(0.0, 1.0, 0.0)) * render_target->m_world;
		for (const auto& camera_pitch:params.cam_pitch) {
			// set camera rotation
			cur_ppc->PositionAndOrient(glm::rotate(deg2rad(camera_pitch), glm::vec3(-1.0, 0.0, 0.0)) * ppc_relative + render_target_center,
				render_target_center - vec3(0.0, render_target_center.y * 0.5f, 0.0),
				vec3(0.0, 1.0, 0.0));
			
			// random sample fovs
			float old_focal = cur_ppc->get_focal();
			float old_dist = glm::length(ppc_relative);

            /**
			float random_fov = pd::normal_random(60,40);
			while(random_fov<=20 || random_fov >=140) {
				random_fov = pd::normal_random(60,40);
			}
            **/
            float random_fov = (float)fov_list[params.model_id % (int)fov_list.size()];

			cur_ppc->set_fov(random_fov);
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
			GC(cudaMalloc(&world_verts_cuda, world_verts.size() * sizeof(glm::vec3)));
			GC(cudaMalloc(&aabb_cuda, sizeof(AABB)));
			GC(cudaMemcpy(world_verts_cuda, world_verts.data(), world_verts.size() * sizeof(vec3), cudaMemcpyHostToDevice));
			GC(cudaMemcpy(aabb_cuda, &aabb, sizeof(AABB), cudaMemcpyHostToDevice));

			// leave enough shadow space
			mask_render(world_verts_cuda, world_verts.size(), aabb_cuda, *cur_ppc, pixels, out_pixels, out_img, "");
			int highest_h = 0;
			for(int i = 0; i < params.w; ++i) {
				for(int j = 0; j < params.h; ++j) {
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
			char buff[100]; snprintf(buff, sizeof(buff), "pitch_%d_rot_%d_fov_%d", (int)camera_pitch, (int)target_rot, (int)random_fov);
			cur_prefix = buff;

			// ------------------------------------ mask ------------------------------------ //
			if (params.render_mask) {
				output_fname = output_folder + "/" + cur_prefix + "_mask.png";
				mask_render(world_verts_cuda, world_verts.size(), aabb_cuda, *cur_ppc, pixels, out_pixels, out_img, output_fname);
			}

			// ------------------------------------ normal ------------------------------------ //
			if(params.render_normal) {
				output_fname = output_folder + "/" + cur_prefix + "_normal.png";
				normal_render(world_verts_cuda, world_verts.size(), aabb_cuda, *cur_ppc, pixels, out_pixels, out_img, output_fname);
			}

			// ------------------------------------ depth ------------------------------------ //
			if(params.render_depth) {
				output_fname = output_folder + "/" + cur_prefix + "_depth.png";
				depth_render(world_verts_cuda, world_verts.size(), aabb_cuda, *cur_ppc, pixels, out_pixels, out_img, output_fname);
			}
			
			// ------------------------------------ touching ------------------------------------ // 
			if (params.render_touch) {
				output_fname = output_folder + "/" + cur_prefix + "_touch.png";

				touch_render(world_verts_cuda, world_verts.size(), aabb_cuda, ground_plane, *cur_ppc, pixels, out_pixels, out_img, output_fname);
			}

			// ------------------------------------ shadow ------------------------------------ // 
			if (params.render_shadow)
				shadow_render(world_verts_cuda, world_verts.size(), ground_plane, aabb_cuda, *cur_ppc, render_target_center,camera_pitch, target_rot, pixels, tmp_pixels, out_pixels, out_img);
			
			cudaFree(world_verts_cuda);
			cudaFree(aabb_cuda);
			*cur_ppc = old_ppc;
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