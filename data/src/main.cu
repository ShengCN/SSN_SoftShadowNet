#include <common.h>
#include <Render_DS/Render_DS.h>
#include <Data_Renderer/Data_Renderer.h>
#include <Utilities/cuda_helper.h>

#include "arg_parse.h"

using namespace purdue;

/* Note different machine will generate different random numbers
 * Use precomputed random numbers insteaad
*/
std::vector<float> fov_list = { 50,44,42,42,47,49,48,44,40,54,48,56,57,51,47,53,59,50,47,48,58,57,43,54,42,44,51,43,47,49,45,51,53,52,56,51,43,43,56,51,42,52,47,42,45,52,46,50,56,48,57,51,54,56,55,53,59,50,44,59,46,48,45,54,56,42,56,   51,49,48,50,49,43,56,45,45,44,40,54,42,50,54,40,54,58,58,43,56,58,48,58,50,44,43,48,59,53,47,41,52};
exp_params params;

/* Rendering Global Variables*/
float light_relative_length = 41.0f;    /* How far is the light */
int counter=0,total_counter=0;          /* Progress */
int i_begin = 256/2;
timer profiling;


std::vector<pixel_pos> ibl_map;

// given an ibl map and light center position
// return the 3D light position
vec3 compute_light_pos(int x, int y, int w = 512, int h = 256) {
	//todo
	float x_fract = (float)x / w, y_fract = (float)y / h;
	deg alpha, beta;
	alpha = x_fract * 360.0f - 90.0f; beta = y_fract * 180.0f - 90.0f;
	return vec3(cos(deg2rad(beta)) * cos(deg2rad(alpha)), sin(deg2rad(beta)), cos(deg2rad(beta)) * sin(deg2rad(alpha)));
}


int main(int argc, char *argv[]) {
	params = parse(argc, argv); 
	create_folder(params.output);

	printf("begin render %s \n", params.model.c_str());
	
	render_data(params.model, params.output);
	return 0;
}
