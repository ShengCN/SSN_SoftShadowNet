#include <string>
#include <glm/gtx/transform.hpp>

#include "model_loader.h"
#include "common.h"
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

void obj_loader::print_info(const tinyobj::attrib_t& attrib, 
							const std::vector<tinyobj::shape_t>& shapes, 
							const std::vector<tinyobj::material_t>& materials) {
	std::cout << "# of vertices  : " << (attrib.vertices.size() / 3) << std::endl;
	std::cout << "# of normals   : " << (attrib.normals.size() / 3) << std::endl;
	std::cout << "# of texcoords : " << (attrib.texcoords.size() / 2)
		<< std::endl;

	std::cout << "# of shapes    : " << shapes.size() << std::endl;
	std::cout << "# of materials : " << materials.size() << std::endl;

	for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
		printf("  v[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
			   static_cast<const double>(attrib.vertices[3 * v + 0]),
			   static_cast<const double>(attrib.vertices[3 * v + 1]),
			   static_cast<const double>(attrib.vertices[3 * v + 2]));
	}

	for (size_t v = 0; v < attrib.normals.size() / 3; v++) {
		printf("  n[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
			   static_cast<const double>(attrib.normals[3 * v + 0]),
			   static_cast<const double>(attrib.normals[3 * v + 1]),
			   static_cast<const double>(attrib.normals[3 * v + 2]));
	}

	for (size_t v = 0; v < attrib.texcoords.size() / 2; v++) {
		printf("  uv[%ld] = (%f, %f)\n", static_cast<long>(v),
			   static_cast<const double>(attrib.texcoords[2 * v + 0]),
			   static_cast<const double>(attrib.texcoords[2 * v + 1]));
	}

	// For each shape
	for (size_t i = 0; i < shapes.size(); i++) {
		printf("shape[%ld].name = %s\n", static_cast<long>(i),
			   shapes[i].name.c_str());
		printf("Size of shape[%ld].mesh.indices: %lu\n", static_cast<long>(i),
			   static_cast<unsigned long>(shapes[i].mesh.indices.size()));
		printf("Size of shape[%ld].lines.indices: %lu\n", static_cast<long>(i),
			   static_cast<unsigned long>(shapes[i].lines.indices.size()));
		printf("Size of shape[%ld].points.indices: %lu\n", static_cast<long>(i),
			   static_cast<unsigned long>(shapes[i].points.indices.size()));

		size_t index_offset = 0;

		assert(shapes[i].mesh.num_face_vertices.size() ==
			   shapes[i].mesh.material_ids.size());

		assert(shapes[i].mesh.num_face_vertices.size() ==
			   shapes[i].mesh.smoothing_group_ids.size());

		printf("shape[%ld].num_faces: %lu\n", static_cast<long>(i),
			   static_cast<unsigned long>(shapes[i].mesh.num_face_vertices.size()));

		// For each face
		for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
			size_t fnum = shapes[i].mesh.num_face_vertices[f];

			printf("  face[%ld].fnum = %ld\n", static_cast<long>(f),
				   static_cast<unsigned long>(fnum));

			// For each vertex in the face
			for (size_t v = 0; v < fnum; v++) {
				tinyobj::index_t idx = shapes[i].mesh.indices[index_offset + v];
				printf("    face[%ld].v[%ld].idx = %d/%d/%d\n", static_cast<long>(f),
					   static_cast<long>(v), idx.vertex_index, idx.normal_index,
					   idx.texcoord_index);
			}

			printf("  face[%ld].material_id = %d\n", static_cast<long>(f),
				   shapes[i].mesh.material_ids[f]);
			printf("  face[%ld].smoothing_group_id = %d\n", static_cast<long>(f),
				   shapes[i].mesh.smoothing_group_ids[f]);

			index_offset += fnum;
		}

		printf("shape[%ld].num_tags: %lu\n", static_cast<long>(i),
			   static_cast<unsigned long>(shapes[i].mesh.tags.size()));
		for (size_t t = 0; t < shapes[i].mesh.tags.size(); t++) {
			printf("  tag[%ld] = %s ", static_cast<long>(t),
				   shapes[i].mesh.tags[t].name.c_str());
			printf(" ints: [");
			for (size_t j = 0; j < shapes[i].mesh.tags[t].intValues.size(); ++j) {
				printf("%ld", static_cast<long>(shapes[i].mesh.tags[t].intValues[j]));
				if (j < (shapes[i].mesh.tags[t].intValues.size() - 1)) {
					printf(", ");
				}
			}
			printf("]");

			printf(" floats: [");
			for (size_t j = 0; j < shapes[i].mesh.tags[t].floatValues.size(); ++j) {
				printf("%f", static_cast<const double>(
					shapes[i].mesh.tags[t].floatValues[j]));
				if (j < (shapes[i].mesh.tags[t].floatValues.size() - 1)) {
					printf(", ");
				}
			}
			printf("]");

			printf(" strings: [");
			for (size_t j = 0; j < shapes[i].mesh.tags[t].stringValues.size(); ++j) {
				printf("%s", shapes[i].mesh.tags[t].stringValues[j].c_str());
				if (j < (shapes[i].mesh.tags[t].stringValues.size() - 1)) {
					printf(", ");
				}
			}
			printf("]");
			printf("\n");
		}
	}

	for (size_t i = 0; i < materials.size(); i++) {
		printf("material[%ld].name = %s\n", static_cast<long>(i),
			   materials[i].name.c_str());
		printf("  material.Ka = (%f, %f ,%f)\n",
			   static_cast<const double>(materials[i].ambient[0]),
			   static_cast<const double>(materials[i].ambient[1]),
			   static_cast<const double>(materials[i].ambient[2]));
		printf("  material.Kd = (%f, %f ,%f)\n",
			   static_cast<const double>(materials[i].diffuse[0]),
			   static_cast<const double>(materials[i].diffuse[1]),
			   static_cast<const double>(materials[i].diffuse[2]));
		printf("  material.Ks = (%f, %f ,%f)\n",
			   static_cast<const double>(materials[i].specular[0]),
			   static_cast<const double>(materials[i].specular[1]),
			   static_cast<const double>(materials[i].specular[2]));
		printf("  material.Tr = (%f, %f ,%f)\n",
			   static_cast<const double>(materials[i].transmittance[0]),
			   static_cast<const double>(materials[i].transmittance[1]),
			   static_cast<const double>(materials[i].transmittance[2]));
		printf("  material.Ke = (%f, %f ,%f)\n",
			   static_cast<const double>(materials[i].emission[0]),
			   static_cast<const double>(materials[i].emission[1]),
			   static_cast<const double>(materials[i].emission[2]));
		printf("  material.Ns = %f\n",
			   static_cast<const double>(materials[i].shininess));
		printf("  material.Ni = %f\n", static_cast<const double>(materials[i].ior));
		printf("  material.dissolve = %f\n",
			   static_cast<const double>(materials[i].dissolve));
		printf("  material.illum = %d\n", materials[i].illum);
		printf("  material.map_Ka = %s\n", materials[i].ambient_texname.c_str());
		printf("  material.map_Kd = %s\n", materials[i].diffuse_texname.c_str());
		printf("  material.map_Ks = %s\n", materials[i].specular_texname.c_str());
		printf("  material.map_Ns = %s\n",
			   materials[i].specular_highlight_texname.c_str());
		printf("  material.map_bump = %s\n", materials[i].bump_texname.c_str());
		printf("    bump_multiplier = %f\n", static_cast<const double>(materials[i].bump_texopt.bump_multiplier));
		printf("  material.map_d = %s\n", materials[i].alpha_texname.c_str());
		printf("  material.disp = %s\n", materials[i].displacement_texname.c_str());
		printf("  <<PBR>>\n");
		printf("  material.Pr     = %f\n", static_cast<const double>(materials[i].roughness));
		printf("  material.Pm     = %f\n", static_cast<const double>(materials[i].metallic));
		printf("  material.Ps     = %f\n", static_cast<const double>(materials[i].sheen));
		printf("  material.Pc     = %f\n", static_cast<const double>(materials[i].clearcoat_thickness));
		printf("  material.Pcr    = %f\n", static_cast<const double>(materials[i].clearcoat_thickness));
		printf("  material.aniso  = %f\n", static_cast<const double>(materials[i].anisotropy));
		printf("  material.anisor = %f\n", static_cast<const double>(materials[i].anisotropy_rotation));
		printf("  material.map_Ke = %s\n", materials[i].emissive_texname.c_str());
		printf("  material.map_Pr = %s\n", materials[i].roughness_texname.c_str());
		printf("  material.map_Pm = %s\n", materials[i].metallic_texname.c_str());
		printf("  material.map_Ps = %s\n", materials[i].sheen_texname.c_str());
		printf("  material.norm   = %s\n", materials[i].normal_texname.c_str());
		std::map<std::string, std::string>::const_iterator it(
			materials[i].unknown_parameter.begin());
		std::map<std::string, std::string>::const_iterator itEnd(
			materials[i].unknown_parameter.end());

		for (; it != itEnd; it++) {
			printf("  material.%s = %s\n", it->first.c_str(), it->second.c_str());
		}
		printf("\n");
	}
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
