#pragma once
#include <common.h>
#include "mesh.h"
#include "tinyobjloader/tiny_obj_loader.h"

enum model_type {
	obj,
	fbx,
	stl,
	off,
	unknown
};

class model_loader
{
public:
	model_loader() = default;
	~model_loader();

public:
	static std::shared_ptr<model_loader> create(model_type mt);
	static std::shared_ptr<model_loader> create(const std::string file_path);

	//------- Interface --------//
public:
	virtual bool load_model(std::string file_path, std::shared_ptr<mesh>& m) = 0;
	virtual bool save_model(std::string file_path, std::shared_ptr<mesh>& m) = 0;

};

class obj_loader : public model_loader {
public:
	obj_loader() = default;
	~obj_loader() {};

	//------- Interface --------//
public:
	virtual bool load_model(std::string file_path, std::shared_ptr<mesh>& m) override;
	virtual bool save_model(std::string file_path, std::shared_ptr<mesh>& m) override;

private:
	void print_info(const tinyobj::attrib_t& attrib,
					const std::vector<tinyobj::shape_t>& shapes,
					const std::vector<tinyobj::material_t>& materials);
};

class off_loader:public model_loader {
public:
	off_loader() = default;
	~off_loader() {};

	//------- Interface --------//
public:
	virtual bool load_model(std::string file_path, std::shared_ptr<mesh>& m) override;
	virtual bool save_model(std::string file_path, std::shared_ptr<mesh>& m) override;
};