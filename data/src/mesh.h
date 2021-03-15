#pragma once

#include <vector>
#include <algorithm>
#include <memory>

#include "ppc.h"

/*!
 * \class Axis aligned bounding box
 *
 * \brief Simplified geometry for fast pruning 
 *
 * \author YichenSheng
 * \date August 2019
 */
struct AABB
{
	vec3 p0, p1;
	AABB(vec3 p) :p0(p), p1(p) {}
	AABB(vec3 sm, vec3 lg) :p0(sm), p1(lg) {} // small, large

	void add_point(vec3 p) {
		p0.x = std::min(p.x, p0.x);
		p0.y = std::min(p.y, p0.y);
		p0.z = std::min(p.z, p0.z);

		p1.x = std::max(p.x, p1.x);
		p1.y = std::max(p.y, p1.y);
		p1.z = std::max(p.z, p1.z);
	}

	void add_aabb(const AABB &new_aabb) {
		add_point(new_aabb.p0);
		add_point(new_aabb.p1);
	}

	float diag_length() {
		return glm::distance(p0, p1);
	}
	
	vec3 diagonal() {
		return p1 - p0;
	}

	vec3 center() {
		return 0.5f * (p1 + p0);
	}
};

/*!
 * \class Base class for mesh
 *
 * \brief 
 *
 * \author YichenSheng
 * \date August 2019
 */
class mesh
{
public:
	mesh();
	~mesh();

	//------- shared functions --------//
public:
	std::vector<vec3> compute_world_space_coords();
	std::vector<vec3> compute_world_space_normals();
	vec3 compute_center();	// center in model space
	vec3 compute_world_center();  // center in world space
	
	void add_world_transate(vec3 v);
	void add_scale(vec3 s);	// scale in model space
	void add_rotate(pd::deg angle, vec3 axis);

	void add_face(vec3 va, vec3 vb, vec3 vc);	// estimate normal
	void add_vertex(vec3 v, vec3 n, vec3 c);
	void add_vertex(vec3 v, vec3 n, vec3 c, vec2 uv);
	AABB compute_aabb() const;
	AABB compute_world_aabb();
	void set_color(vec3 col);

	void normalize_position_orientation(vec3 scale=vec3(1.0f), 
										glm::quat rot_quant = glm::quat(0.0f,0.0f,0.0f,0.0f));
	
	void get_demose_matrix(vec3& scale, quat& rot, vec3& translate);
	void set_matrix(const vec3 scale, const quat rot, const vec3 translate);
	void clear_vertices() { m_world = glm::identity<mat4>(); m_verts.clear(); m_norms.clear(); m_colors.clear(); m_uvs.clear(); }
	void recompute_normal();
	void remove_duplicate_vertices();
	std::string to_string() {
		return std::to_string(get_id());
	}

	//------- getter & setter --------//
public:
	mat4 get_world_mat() { return m_world; }
	void set_world_mat(const mat4 m) { m_world = m; }
	int get_id() { return cur_id; }
	bool get_is_selected() { return m_is_selected; }
	void set_is_selected(bool is_selected) { m_is_selected = is_selected; }
	void set_light(bool trigger = true) { m_is_emitter = true; }
	bool is_light() { return m_is_emitter; }

	//------- member variables --------//
public:
	mat4 m_world = glm::mat4(1.0f); // model space -> world space
	std::vector<vec3> m_verts;
	std::vector<vec3> m_norms;
	std::vector<vec3> m_colors;
	std::vector<vec2> m_uvs;
	std::string file_path;
	
	std::string m_vs, m_fs;
	int cur_id = -1;
	static int id;
	bool m_is_selected = false;
	bool m_is_emitter = false;
};