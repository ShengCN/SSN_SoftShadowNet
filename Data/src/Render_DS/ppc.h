#pragma once
#include <common.h>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

struct ray {
	vec3 ro, rd;
};

/*
 *	Planer Pinhole camera, basic controls.
 */
enum class CameraMovement {
	forward,
	backward,
	left,
	right,
	up,
	down
};

class ppc {
public:
	glm::vec3 _position;
	glm::vec3 _front;
	glm::vec3 _up;	// default is (0.0f,1.0f,0.0f)
	float _fov, _near, _far;
	int _width, _height;

	ppc()=default;
	ppc(int w, int h, float fov, float p_near=2.0f, float p_far=1000.0f);
	~ppc();

	bool save(const std::string file);
	bool load(const std::string file);

	CUDA_HOSTDEV
	vec3 GetRight() const {
        vec3 view = GetViewVec();
        return glm::normalize(cross(view, _up));
    }
	
    CUDA_HOSTDEV
    vec3 GetUp() const {
        return glm::normalize(cross(GetRight(), GetViewVec()));
    }

    CUDA_HOSTDEV
	vec3 GetViewVec() const { return glm::normalize(_front); }
	
	CUDA_HOSTDEV
	float get_fov() const { return _fov; }

	CUDA_HOSTDEV
	void set_fov(float f) { _fov = f;}
    
	CUDA_HOSTDEV
    void caliberate_same(float d1, glm::vec3 center) {
        /* Given d1, calibrate fov to keep current distance the same as d1 
         * a/f * d = c 
         * 
         * */
        float cur_dis = glm::distance(center, _position);
        float focal = get_focal();
        float new_focal = focal / cur_dis * d1;

        /* focal -> fov*/
        set_fov(2.0f * purdue::rad2deg(atan2(height()*0.5f, new_focal)));
    }

	CUDA_HOSTDEV
    void set_aa_sample(size_t aasp, size_t aasi, size_t aasj) {
        m_aasp = aasp;
        m_aasi = aasi;
        m_aasj = aasj;
    }

	CUDA_HOSTDEV
	void PositionAndOrient(vec3 p, vec3 lookatP, vec3 up){
        _position = p;
        _front = glm::normalize(lookatP - p);
        _up = up;
    }
    
	CUDA_HOSTDEV
	void look_at(vec3 center){
        _front = glm::normalize(center - _position);
    }

   	CUDA_HOSTDEV
	glm::mat4 GetP() const {
		return glm::perspective(purdue::deg2rad(_fov), (float)_width / (float)_height, _near, _far);
	}

   	CUDA_HOSTDEV
	glm::mat4 GetV() const {
		return glm::lookAt(_position, _position + _front, _up);
	}

	glm::mat3 get_local_mat();
	void Rotate_Axis(glm::vec3 O, glm::vec3 axis, float angled);	// only rotate ppc position
	void Zoom(float delta);
	void Keyboard(CameraMovement cm, float speed); // keyboard reactions
	void scroll(double delta) {
		_fov += float(delta);
	}

	vec3 get_pos() { return _position; }
	void pan(double deg);
	void tilt(double deg);
	void pitch(double deg);
	void mouse_press(int x, int y);
	void mouse_release(int x, int y);
	void mouse_move(int x, int y);
	bool mouse_inside_frame(int x, int y);
	void set_trackball(bool trackball);
	void camera_resize(int w, int h);
	int width() { return _width; }
	int height() { return _height; }
    void set_size(int w, int h) {_width = w; _height = h;}
	float GetFocal();

   	CUDA_HOSTDEV
	vec3 project(vec3 p) {
        /*
         * return <u, v, t> 
         * */
		vec3 a = glm::normalize(GetRight());
		vec3 b = -glm::normalize(GetUp());
		vec3 c = glm::normalize(GetViewVec()) * get_focal() - 0.5f * (float)_width * GetRight() + 0.5f * (float)_height * GetUp();
		mat3 m(a,b,c);
		vec3 pp = glm::inverse(m) * (p-_position);

        vec2 pix(pp.x/pp.z, _height-pp.y/pp.z);
        vec3 rd = m * vec3(pix, 1.0f);
		return vec3(pix, pp.z * sqrtf(glm::dot(rd, rd)));
	}

   	CUDA_HOSTDEV
	ray get_ray(float u, float v) const {
		ray ret;
        float focal = get_focal();
        vec3 right = glm::normalize(GetRight());
        vec3 up = glm::normalize(GetUp());
        vec3 front = glm::normalize(GetViewVec());

        float delta_aa = 1.0/((float)m_aasp + 1.0);
        ret.ro = _position;
        int center_u = _width / 2, center_v = _height / 2;

        ret.rd = front * focal + 
            (u - center_u + delta_aa * (float)(m_aasi + 1)) * right + 
            (v - center_v + delta_aa * (float)(m_aasj + 1)) * up;

        ret.rd = glm::normalize(ret.rd);
		return ret;
    }

   CUDA_HOSTDEV
    float get_focal() const {
        float rad = 0.5f * _fov /180.0 * 3.14159265;
        return 0.5f * _height/ std::tan(rad);
    }

	std::string to_string();

private:
	int m_last_x, m_last_y;
	vec3 m_last_orientation, m_last_position;
	bool m_pressed;
	bool m_trackball;
    size_t m_aasp, m_aasi, m_aasj;
};
