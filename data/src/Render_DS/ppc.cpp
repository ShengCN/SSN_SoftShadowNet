#include "ppc.h"

using namespace glm;
using namespace purdue;

float ppc::GetFocal() {
	return static_cast<float>(_width / 2) / tan(_fov / 2.0f);
}

ppc::ppc(int w, int h, float fov, float p_near, float p_far) :
	_width(w),
	_height(h),
	_fov(fov),
	_near(p_near),
	_far(p_far),
	_position(0.0f, 0.35f, 1.3f), 
	_up(0.0f, 1.0f, 0.0f),
	m_pressed(false),
	m_trackball(true), 
    m_aasp(2),
    m_aasi(0),
    m_aasj(0) {
}

ppc::~ppc() {
}

bool ppc::save(const std::string file) {
	std::ofstream output(file, std::ofstream::binary);
	if (output.is_open()) {
		output.write((char*)&_position[0], sizeof(glm::vec3));
		output.write((char*)&_front[0], sizeof(glm::vec3));
		output.write((char*)&_up[0], sizeof(glm::vec3));
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

bool ppc::load(const std::string file) {
	std::ifstream input(file, std::ifstream::binary);
	if (input.is_open()) {
		input.read((char*)&_position[0], sizeof(glm::vec3));
		input.read((char*)&_front[0], sizeof(glm::vec3));
		input.read((char*)&_up[0], sizeof(glm::vec3));
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

glm::mat3 ppc::get_local_mat() {
	vec3 x = GetRight(), y = GetUp(), z = -GetViewVec();
	return glm::mat3(x, y, z);
}

void ppc::Rotate_Axis(glm::vec3 O, glm::vec3 axis, float angled)
{
	glm::vec3 CO = _position - O;
	auto rot_mat = glm::rotate(deg2rad(angled), normalize(axis));
	CO = vec3(rot_mat * vec4(CO, 0.0f));
	_position = CO + O;

}

void ppc::Zoom(float delta)
{
	_fov += delta;
	_fov = glm::clamp(_fov, 10.0f, 150.0f);
}

void ppc::Keyboard(CameraMovement cm, float speed)
{
	switch (cm)
	{
	case CameraMovement::forward:
		_position += speed * _front;
		break;
	case CameraMovement::backward:

		_position -= speed * _front;
		break;
	case CameraMovement::left:

		_position -= speed * GetRight();
		break;
	case CameraMovement::right:
		_position += speed * GetRight();
		break;
	case CameraMovement::up:
		_position += speed * GetUp();
		break;
	case CameraMovement::down:
		_position -= speed * GetUp();
		break;
	default:
		break;
	}

	//INFO("Current PPC: ");
	//INFO(this->to_string());	
}

void ppc::pan(double deg)
{
	deg = deg / _width * 10.0f;
	//glm::mat4 rot_y = glm::rotate(deg2rad(deg), glm::vec3(0.0f, 1.0f, 0.0f));
	//_front = glm::vec3(rot_y * glm::vec4(_front, 0.0f));
	vec3 right = GetRight();
	_front += right * float(deg);
	_front = glm::normalize(_front);
}

void ppc::tilt(double deg)
{
	// #TODO
}

void ppc::pitch(double deg)
{
	//deg = deg / _height * 10.0f;
	//vec3 up = GetUp();
	//_front += up * float(deg);
	//_front = glm::normalize(_front);

    vec3 right = glm::cross(_front, vec3(0.0f,1.0f,0.0f));
	vec4 tmp = glm::rotate(deg2rad(deg), right) * vec4(_front, 0.0f);
	_front = glm::normalize(vec3(tmp));
}

void ppc::mouse_press(int x, int y) {
	m_last_x = x; m_last_y = y;
	m_last_orientation = _front;
	m_last_position = _position;
	m_pressed = true;
}

void ppc::mouse_release(int x, int y) {
	m_pressed = false;
	
	m_last_x = x; m_last_y = y;
	m_last_orientation = _front;
	m_last_position = _position;	
}


void ppc::mouse_move(int x, int y) {
	if (!m_pressed || std::isnan(x))
		return;
	//x = pd::clamp(x, 0, _width);
	//y = pd::clamp(y, 0, _height);

	auto arcball_vector=[](int x, int y, int w, int h){
		float x_fract = (float)x / w * 2.0f - 1.0f, y_fract = 1.0f - (float)y / h * 2.0f;
		return glm::normalize(vec3(x_fract, y_fract, std::sqrt(1.0f - x_fract * x_fract - y_fract * y_fract) ));
	};

	vec3 last = arcball_vector(m_last_x, m_last_y, _width, _height), cur = arcball_vector(x, y, _width, _height);
	
	float dot_ang = glm::dot(last, cur);
	//dot_ang = pd::clamp(dot_ang, -1.0f, 1.0f);
	// INFO("Last/Cur: {},{}", pd::to_string(last), pd::to_string(cur));
	vec3 rot_axis = glm::normalize(glm::cross(last, cur)); rad rot_ang = std::acos(std::min(dot_ang,1.0f));
	rot_axis = get_local_mat() * rot_axis;

	if(glm::all(glm::isnan(rot_axis))) {
		return;
	}	

	float speed = 0.75f;
	rot_ang = rot_ang * speed;
	
	if(!m_trackball) {
		_front = glm::vec3(glm::rotate(rot_ang, rot_axis) * vec4(m_last_orientation,0.0f));
		// INFO(pd::to_string(relative));
		// _front = glm::normalize(glm::normalize(m_last_orientation) + relative);
	}
	else {
		// use trackball
		vec4 new_pos = glm::rotate(rot_ang, rot_axis) * vec4(m_last_position, 1.0f);
		m_last_orientation = vec3(new_pos/new_pos.w);
		PositionAndOrient(m_last_orientation, vec3(0.0f), _up);

		/* DBG */
		// INFO("Rotation Axis: {}, Rotation Angle: {}", purdue::to_string(rot_axis), rot_ang);
		// INFO("Last Position/New Pos, {}, {}", purdue::to_string(m_last_position), purdue::to_string(m_last_orientation));
	}
}

void ppc::set_trackball(bool tb) {
	m_trackball = tb;
}


void ppc::camera_resize(int w, int h) {
	_width = w;
	_height = h;
}

std::string ppc::to_string() {
	std::ostringstream oss;
	oss << "w: " << _width << " h: " << _height<< std::endl;
	oss << "look: " << purdue::to_string(_position) << std::endl;
	oss << "at: " << purdue::to_string(_position + _front) << std::endl;
	oss << "up: " << purdue::to_string(GetUp()) << std::endl;
	oss << "fov: " << std::to_string(_fov) << std::endl;

	return oss.str();
}
