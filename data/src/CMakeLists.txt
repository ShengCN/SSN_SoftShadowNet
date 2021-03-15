cmake_minimum_required (VERSION 3.8)
project (hard_shadow LANGUAGES CXX CUDA)

enable_language(CUDA)
include_directories(${PROJECT_SOURCE_DIR})

add_library(tinyobj ${PROJECT_SOURCE_DIR}/tinyobjloader/tiny_obj_loader.cc)

set (CMAKE_CXX_STANDARD 11)

add_executable(hard_shadow main.cu common.cpp mesh.cpp model_loader.cpp ppc.cpp)
add_executable(hard_shadow_fov force_fov.cu common.cpp mesh.cpp model_loader.cpp ppc.cpp)

target_link_libraries (hard_shadow PRIVATE tinyobj)
target_link_libraries (hard_shadow_fov PRIVATE tinyobj)

# target_compile_options(hard_shadow PRIVATE "${OpenMP_CXX_FLAGS}")

set_property(TARGET hard_shadow 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET hard_shadow_fov
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)