#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <memory>

#include <common.h>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

/*
 *
 *	Planer Pinhole camera.
 *	Provide basic controls.
 *
 */
enum class CameraMovement
{
	forward,
	backward,
	left,
	right,
	up,
	down
};

struct ray {
	glm::vec3 ro, rd;
};

class ppc
{
public:
	glm::vec3 _position;
	glm::vec3 _front;
	glm::vec3 _worldUp;	// default is (0.0f,1.0f,0.0f)
	float _fov, _near, _far;
	int _width, _height;

	ppc(int w, int h, float fov, float p_near=0.001f, float p_far=100.0f);
	~ppc();

	bool save(const std::string file) {
		std::ofstream output(file, std::ofstream::binary);
		if (output.is_open()) {
			output.write((char*)&_position[0], sizeof(glm::vec3));
			output.write((char*)&_front[0], sizeof(glm::vec3));
			output.write((char*)&_worldUp[0], sizeof(glm::vec3));
			output.write((char*)&_fov, sizeof(float));
			output.write((char*)&_near, sizeof(float));
			output.write((char*)&_far, sizeof(float));
			output.write((char*)&_width, sizeof(int));
			output.write((char*)&_height, sizeof(int));
			output.close();
			return true;
		}
		else {
			std::cout << "File " << file << " cannot save.\n";
			return false;
		}
	}

	bool load(const std::string file) {
		std::ifstream input(file, std::ifstream::binary);
		if (input.is_open()) {
			input.read((char*)&_position[0], sizeof(glm::vec3));
			input.read((char*)&_front[0], sizeof(glm::vec3));
			input.read((char*)&_worldUp[0], sizeof(glm::vec3));
			input.read((char*)&_fov, sizeof(float));
			input.read((char*)&_near, sizeof(float));
			input.read((char*)&_far, sizeof(float));
			input.read((char*)&_width, sizeof(int));
			input.read((char*)&_height, sizeof(int));
			input.close();
			return true;
		}
		else {
			std::cout << "File " << file << " cannot save.\n";
			return false;
		}
	}
   
    CUDA_HOSTDEV
	vec3 GetRight() const {
        vec3 view = GetViewVec();

        return cross(view, _worldUp);
    }
	
    CUDA_HOSTDEV
    vec3 GetUp() const {
        return cross(GetRight(), GetViewVec());
    }
    CUDA_HOSTDEV
	vec3 GetViewVec() const { return _front; }
	
	CUDA_HOSTDEV
	float get_fov() const { return _fov; }

	CUDA_HOSTDEV
	void set_fov(float f) { _fov = f;}
    
    CUDA_HOSTDEV
	void PositionAndOrient(vec3 p, vec3 lookatP, vec3 up){
        _position = p;
        _front = glm::normalize(lookatP - p);
        _worldUp = up;
    }
    
	glm::mat4 GetP() const;
	glm::mat4 GetV() const;
	void Rotate_Axis(glm::vec3 O, glm::vec3 axis, float angled);	// only rotate ppc position
	void Zoom(float delta);
	void Keyboard(CameraMovement cm, float speed); // keyboard reactions
	void scroll(double delta) {
		_fov += float(delta);
	}

	void pan(double deg);
	void tilt(double deg);
	void pitch(double deg);
    
   CUDA_HOSTDEV
	void get_ray(int u, int v, vec3& ro, vec3& rd) const {
        float focal = get_focal();
        vec3 right = glm::normalize(GetRight());
        vec3 up = glm::normalize(GetUp());
        vec3 front = glm::normalize(GetViewVec());

        ro = _position;
        int center_u = _width / 2, center_v = _height / 2;

        rd = front * focal + (u - center_u + 0.5f) * right + (v - center_v + 0.5f) * up;
        rd = glm::normalize(rd);
    }

   CUDA_HOSTDEV
    float get_focal() const {
        float rad = 0.5f * _fov /180.0 * 3.1415926;
        return 0.5f * _height/ std::tan(rad);
    }

	std::string to_string();

private:
	float GetFocal() const;
};
