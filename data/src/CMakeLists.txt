cmake_minimum_required (VERSION 3.8)
project (shadow_base LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 11)


# find_package(glfw3 3.3. REQUIRED)
set (CMAKE_CXX_STANDARD 11)
find_package(CUDA  REQUIRED)
enable_language(CUDA)

# Dependencies
add_subdirectory(Dep/glm)
add_subdirectory(Dep/fmt)
add_subdirectory(Dep/tinyobjloader)

file(GLOB src
    "${PROJECT_SOURCE_DIR}/*.cu"
    "${PROJECT_SOURCE_DIR}/*.cpp"
)

add_executable(shadow_base ${src}
)

target_include_directories(shadow_base PUBLIC 
${PROJECT_SOURCE_DIR} 
${CUDA_INCLUDE_DIRS} 
${PROJECT_SOURCE_DIR}/Dep/fmt/include 
${PROJECT_SOURCE_DIR}/Dep/glm 
${PROJECT_SOURCE_DIR}/Dep
)

target_link_libraries(shadow_base PUBLIC
glm
fmt
tinyobjloader
)

set_property(TARGET shadow_base 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
