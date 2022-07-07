#include "model_loader.h"
#include "tinyobjloader/tiny_obj_loader.h"
using namespace purdue;

model_loader::~model_loader() {
}

std::shared_ptr<model_loader> model_loader::create(model_type mt) {
	std::shared_ptr<model_loader> ret;
	switch (mt) {
	case obj:
		ret = std::make_shared<obj_loader>();
		break;
	case off:
		ret = std::make_shared<off_loader>();
		break;

	default:
		ret = std::make_shared<obj_loader>();
		break;
	}

	return ret;
}

std::shared_ptr<model_loader> model_loader::create(const std::string file_path) {
	std::string ext = get_file_ext(file_path);
	if(ext == "obj") {
		return create(model_type::obj);
	} else if(ext == "off") {
		return create(model_type::off);
	}
	return create(model_type::unknown);
}

bool obj_loader::load_model(std::string file_path, std::shared_ptr<mesh>& m) {
	if(!m) {
		m = std::make_shared<mesh>();
	}
	m->file_path = file_path;
	m->clear_vertices();

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;
	std::string base_path = get_folder(file_path);
	bool triangulate = true;
	
	timer t;
	t.tic();
	bool ret = tinyobj::LoadObj(&attrib,
								&shapes,
								&materials,
								&warn,
								&err,
								file_path.c_str(),
								base_path.c_str(),
								triangulate);
	t.toc();
	t.print_elapsed();

	if (!warn.empty()) {
		std::cout << "WARN: " << warn << std::endl;
	}

	if (!err.empty()) {
		std::cerr << "ERR: " << err << std::endl;
	}

	if (!ret) {
		printf("Failed to load/parse .obj.\n");
		return false;
	}

	std::cerr << "Success fully loaded file: \n" << file_path << std::endl;

	// For each shape
	for (size_t i = 0; i < shapes.size(); i++) {
		size_t index_offset = 0;

		// For each face
		for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
			size_t fnum = shapes[i].mesh.num_face_vertices[f];

			// For each vertex in the face
			for (size_t v = 0; v < fnum; v++) {
				tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
				glm::vec3 vertex(attrib.vertices[3 * idx.vertex_index],
								 attrib.vertices[3 * idx.vertex_index + 1],
								 attrib.vertices[3 * idx.vertex_index + 2]);

				glm::vec3 normal(attrib.normals[3 * idx.normal_index],
								 attrib.normals[3 * idx.normal_index + 1],
								 attrib.normals[3 * idx.normal_index + 2]);

				m->m_verts.push_back(vertex);
				m->m_norms.push_back(normal);

				if(attrib.texcoords.size()>0){
					glm::vec2 uv(attrib.texcoords[2 * idx.texcoord_index],
								 attrib.texcoords[2 * idx.texcoord_index + 1]);
					m->m_uvs.push_back(uv);
				}
			}

			index_offset += fnum;
		}
	}

	return true;
}

bool obj_loader::save_model(std::string file_path, std::shared_ptr<mesh>& m_) {
	return false;
}

bool off_loader::load_model(std::string file_path, std::shared_ptr<mesh>& m) {
	if(!m) {
		m = std::make_shared<mesh>();
	}

	m->file_path = file_path;
	std::ifstream input(file_path, std::ifstream::in);
	int vert_num, tri_num;
	std::string tmp;
	input >> tmp;
	if (tmp.size() == 3) {
		input >> tmp; vert_num = std::stoi(tmp);
		input >> tmp; tri_num = std::stoi(tmp);
	}
	else {
		vert_num = std::stoi(tmp.substr(3));
		input >> tmp; tri_num = std::stoi(tmp);
	}
	input >> tmp;

	auto split_str = [](const std::string &s) {
		std::vector<std::string> ret;

		std::string cur_str;
		for (auto c : s) {
			if (c == ' ') {
				ret.push_back(cur_str);
				cur_str.clear();
			}
			else
				cur_str.push_back(c);
		}
		ret.push_back(cur_str);
		return ret;
	};

	std::vector<glm::vec3> vertices(vert_num);
	std::vector<int> faces(tri_num * 3);

	for (int i = 0; i < vert_num; ++i) {
		glm::vec3 p;
		input >> tmp; p.x = std::stof(tmp);
		input >> tmp; p.y = std::stof(tmp);
		input >> tmp; p.z = std::stof(tmp);
		vertices[i] = p;
	}

	for (int i = 0; i < tri_num; ++i) {
		input >> tmp;
		input >> tmp; faces[3 * i + 0] = std::stoi(tmp);
		input >> tmp; faces[3 * i + 1] = std::stoi(tmp);
		input >> tmp; faces[3 * i + 2] = std::stoi(tmp);
	}

	m->clear_vertices();
	// translate data to our data structure
	for (auto &f : faces) {
		m->m_verts.push_back(vertices[f]);
	}

	m->recompute_normal();
	return true;
}

bool off_loader::save_model(std::string file_path, std::shared_ptr<mesh>& m) {
	if (!m) {
		return false;
	}

	std::ofstream output(file_path);
	if(output.is_open()) {
		output << "OFF" << std::endl;

		auto &verts = m->m_verts;
		int v_num = verts.size();
		int f_num = v_num / 3;

		output << v_num << " " << f_num << " " << 0 << std::endl;
		for(auto &v:verts) {
			output << v.x << " " << v.y << " " << v.z << std::endl;
		}
		for(int i = 0; i < f_num; ++i) {
			output << 3 << " " << 3 * i + 0 << " " << 3 * i + 1 << " " << 3 * i + 2 << std::endl;
		}

	} else {
		return false;
	}


	output.close();
	return true;
}

bool load_mesh(const std::string fname, std::shared_ptr<mesh> &m) {
	auto loader = model_loader::create(fname);
	return loader->load_model(fname, m);
}