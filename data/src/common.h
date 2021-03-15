#pragma once
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtx/quaternion.hpp>
#include <string>
#include <chrono>
#include <random>
#include <sstream>
#include <iostream>
#include <unistd.h>

typedef std::chrono::high_resolution_clock Clock;

namespace purdue {
	typedef float deg;
	typedef float rad;
	constexpr float pi = 3.14159265f;

	// only for vector p
	inline glm::vec3 operator*(glm::mat4 m, glm::vec3 p) {
		return glm::vec3(m * glm::vec4(p, 1.0f));
	}
};

namespace pd = purdue;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::quat;
using pd::operator *;

//------- Singleton Class for Shared Variables --------//
// src: https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
class shared_variables {
//------- variables --------//
public:
	const std::string shader_folder = "Shaders/";
	const std::string template_vs = shader_folder + "template_vs.glsl";
	const std::string template_fs = shader_folder + "template_fs.glsl";
	vec3 default_stl_color = glm::vec3(0.3f);

public:
	static shared_variables& instance() {
		static shared_variables instance;
		return instance;
	}

private:
	shared_variables() = default;

//------- constraints on copy and move --------//
public:
	shared_variables(shared_variables const&) = delete;
	void operator=(shared_variables const&) = delete;
};


#ifndef GGV	// graphics lib global variables
#define GGV shared_variables::instance()
#endif // !GGV

namespace purdue {
	bool save_image(const std::string fname, unsigned int *pixels, int w, int h, int c = 4);
	
	/*!
	 *
	 * \brief Math lab
	 *
	 * \author YichenSheng
	 * \date August 2019
	 */
	inline pd::rad deg2rad(pd::deg d) {
		return d / 180.0f * pd::pi;
	}

	inline pd::deg rad2deg(pd::rad r) {
		return r / pd::pi * 180.0f;
	}

	inline float deg2quat(pd::deg d) {
		return std::cos(deg2rad(d / 2));
	}
	
	// static std::random_device rd;
	static std::mt19937 engine(0);
	
	inline float normal_random(float center, float sigma) {
		std::normal_distribution<float> dist(center, sigma);
		return dist(engine);
	}

	inline float random_float(float fmin = 0.0f, float fmax = 1.0f) {
		// engine
		// std::random_device rd;
		// std::mt19937 engine(rd());
		std::uniform_real_distribution<float> u(fmin, fmax);
		return u(engine);
	}

	inline int random_int(int mi = 0, int ma = 10) {
		// engine
		// std::random_device rd;
		// std::mt19937 engine(rd());
		std::uniform_int_distribution<> dis(mi, ma);

		return dis(engine);
	}


	template<typename T>
	T lerp(T a, T b, float fract) {
		return (1.0f - fract) * a + fract * b;
	}

	class timer {
	public:
		timer() = default;
		~timer() {};

		void tic() {
			_is_ticed = true;
			_tic = Clock::now();
		}
		void toc() {
			_toc = Clock::now();
		}

		long long get_elapse() {
			return std::chrono::duration_cast<std::chrono::nanoseconds>(_toc - _tic).count();
		}

		void print_elapsed() {
			if (!_is_ticed) {
				std::cerr << "timer has not been ticed \n";
				return;
			}

			auto elapsed = get_elapse();
			std::cerr << "Time: " << elapsed * 1e-9 << " seconds \n";
		}

		std::string to_string() {
			if (!_is_ticed) {
				std::cerr << "timer has not been ticed. \n";
			}
			std::stringstream oss;
			oss << get_elapse() * 1e-9;
			return oss.str();
		}

	private:
		bool _is_ticed = false;
		std::chrono::time_point<std::chrono::system_clock> _tic;
		std::chrono::time_point<std::chrono::system_clock> _toc;
	};

	inline vec3 bary_centric_interpolate(vec3 a, vec3 b, vec3 c, vec3 bary_coord) {
		return a * bary_coord.x + b * bary_coord.y + c * bary_coord.z;
	}

	inline bool float_equal(float a, float b, float eps = 1e-7) {
		return std::abs(a - b) < eps;
	}

	inline std::ostream& operator<<(std::ostream& out, vec3 v) {
		out << "(" << v.x << "," << v.y << "," << v.z << ")";
		return out;
	}

	inline std::string to_string(vec3 v) {
		std::stringstream out;
		out << v.x << "," << v.y << "," << v.z;
		return out.str();
	}

	inline std::string to_string(vec4 v) {
		std::stringstream out;
		out << v.x << "," << v.y << "," << v.z << "," <<v.w;
		return out.str();
	}

	inline std::string to_string(mat4 m) {
		std::stringstream out;
		out << to_string(m[0]) << std::endl;
		out << to_string(m[1]) << std::endl;
		out << to_string(m[2]) << std::endl;
		out << to_string(m[3]);
		return out.str();
	}

	inline bool same_point(const vec3 &a, const vec3 &b) {
		return glm::distance(a, b) < 1e-3;
	}

	inline std::string get_folder(const std::string& str) {
		size_t found;
		found = str.find_last_of("/\\");

		return str.substr(0, found);
	}

	inline std::string get_filename(const std::string& str) {
		size_t found;
		found = str.find_last_of("/\\");

		return str.substr(found + 1);
	}

	inline std::string get_file_ext(const std::string& str) {
		size_t found;
		found = str.find_last_of(".");

		return str.substr(found + 1);
	}

	inline void create_folder(const std::string& folder) {
		std::string command = "mkdir " + folder;
		system(command.c_str());
	}
    
    inline bool exists_test (const std::string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
    }
}
