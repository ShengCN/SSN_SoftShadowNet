#pragma once
#include <common.h>

std::vector<vec2> uniform_sphere_2d_samples(int n) {
    std::vector<vec2> ret(n);
    for(int i = 0;i < n; ++i) {
        float x = pd::random_float();
        float y = pd::random_float();
        ret[i] = vec2(std::acos(std::sqrt(1.0f - x)), 2.0f * pd::pi * y);
    }
    return ret;
}

std::vector<vec3> uniform_sphere_3d_samples(int n) {
    auto samples = uniform_sphere_2d_samples(n);
    std::vector<vec3> ret(n);

    for(int i = 0; i < n; ++i) {
        float a = samples[i].x, b = samples[i].y;
        vec3 p(std::sin(a) * std::cos(b), std::cos(a), std::sin(a) * std::sin(b));
		ret[i] = p;
    }
    return ret;
}
