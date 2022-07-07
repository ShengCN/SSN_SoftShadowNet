#include <common.h>
#include <Data_Renderer/Data_Renderer.h>
#include <Render_DS/Render_DS.h>

int main(int argc, char *argv[]) {
	exp_params params = parse(argc, argv); 
	purdue::create_folder(params.output);
	
	printf("begin render %s \n", params.model.c_str());
	
	render_data(params);
	return 0;
}
