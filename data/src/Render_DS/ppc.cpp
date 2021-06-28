#include "ppc.h"

using namespace glm;
using namespace purdue;

float ppc::GetFocal() const {
	return static_cast<float>(_width / 2) / tan(purdue::deg2rad(_fov / 2.0f));
}

ppc::ppc(int w, int h, float fov, float p_near, float p_far) :
	_width(w),
	_height(h),
	_fov(fov),
	_near(p_near),
	_far(p_far),
	_position(0.0f, 0.35f, 1.3f), 
	_worldUp(0.0f, 1.0f, 0.0f) {
}

ppc::~ppc() {
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

void ppc::tilt(double deg) {
	// #TODO
}

void ppc::pitch(double deg) {
	vec3 right = glm::cross(_front, _worldUp);
	vec4 tmp = glm::rotate(deg2rad(deg), right) * vec4(_front, 0.0f);
	_front = glm::normalize(vec3(tmp));
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
