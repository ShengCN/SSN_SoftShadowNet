# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build

# Include any dependencies generated for this target.
include CMakeFiles/shadow_base.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/shadow_base.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/shadow_base.dir/flags.make

CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.o: CMakeFiles/shadow_base.dir/flags.make
CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.o: ../Data_Renderer/data_rasterizer.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Data_Renderer/data_rasterizer.cu -o CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.o

CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.o: CMakeFiles/shadow_base.dir/flags.make
CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.o: ../Data_Renderer/ray_intersect.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Data_Renderer/ray_intersect.cu -o CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.o

CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.o: CMakeFiles/shadow_base.dir/flags.make
CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.o: ../Data_Renderer/renderer.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Data_Renderer/renderer.cu -o CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.o

CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/shadow_base.dir/Render_DS/Image.cu.o: CMakeFiles/shadow_base.dir/flags.make
CMakeFiles/shadow_base.dir/Render_DS/Image.cu.o: ../Render_DS/Image.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/shadow_base.dir/Render_DS/Image.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Render_DS/Image.cu -o CMakeFiles/shadow_base.dir/Render_DS/Image.cu.o

CMakeFiles/shadow_base.dir/Render_DS/Image.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/shadow_base.dir/Render_DS/Image.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/shadow_base.dir/Render_DS/Image.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/shadow_base.dir/Render_DS/Image.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.o: CMakeFiles/shadow_base.dir/flags.make
CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.o: ../Render_DS/mesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.o -c /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Render_DS/mesh.cpp

CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Render_DS/mesh.cpp > CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.i

CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Render_DS/mesh.cpp -o CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.s

CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.o: CMakeFiles/shadow_base.dir/flags.make
CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.o: ../Render_DS/model_loader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.o -c /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Render_DS/model_loader.cpp

CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Render_DS/model_loader.cpp > CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.i

CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Render_DS/model_loader.cpp -o CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.s

CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.o: CMakeFiles/shadow_base.dir/flags.make
CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.o: ../Render_DS/ppc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.o -c /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Render_DS/ppc.cpp

CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Render_DS/ppc.cpp > CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.i

CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/Render_DS/ppc.cpp -o CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.s

CMakeFiles/shadow_base.dir/arg_parse.cpp.o: CMakeFiles/shadow_base.dir/flags.make
CMakeFiles/shadow_base.dir/arg_parse.cpp.o: ../arg_parse.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/shadow_base.dir/arg_parse.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/shadow_base.dir/arg_parse.cpp.o -c /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/arg_parse.cpp

CMakeFiles/shadow_base.dir/arg_parse.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/shadow_base.dir/arg_parse.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/arg_parse.cpp > CMakeFiles/shadow_base.dir/arg_parse.cpp.i

CMakeFiles/shadow_base.dir/arg_parse.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/shadow_base.dir/arg_parse.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/arg_parse.cpp -o CMakeFiles/shadow_base.dir/arg_parse.cpp.s

CMakeFiles/shadow_base.dir/main.cu.o: CMakeFiles/shadow_base.dir/flags.make
CMakeFiles/shadow_base.dir/main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CUDA object CMakeFiles/shadow_base.dir/main.cu.o"
	/usr/local/cuda-11.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/main.cu -o CMakeFiles/shadow_base.dir/main.cu.o

CMakeFiles/shadow_base.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/shadow_base.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/shadow_base.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/shadow_base.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target shadow_base
shadow_base_OBJECTS = \
"CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.o" \
"CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.o" \
"CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.o" \
"CMakeFiles/shadow_base.dir/Render_DS/Image.cu.o" \
"CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.o" \
"CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.o" \
"CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.o" \
"CMakeFiles/shadow_base.dir/arg_parse.cpp.o" \
"CMakeFiles/shadow_base.dir/main.cu.o"

# External object files for target shadow_base
shadow_base_EXTERNAL_OBJECTS =

CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.o
CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.o
CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.o
CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/Render_DS/Image.cu.o
CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.o
CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.o
CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.o
CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/arg_parse.cpp.o
CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/main.cu.o
CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/build.make
CMakeFiles/shadow_base.dir/cmake_device_link.o: Dep/fmt/libfmtd.a
CMakeFiles/shadow_base.dir/cmake_device_link.o: Dep/tinyobjloader/libtinyobjloader.a
CMakeFiles/shadow_base.dir/cmake_device_link.o: CMakeFiles/shadow_base.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CUDA device code CMakeFiles/shadow_base.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/shadow_base.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/shadow_base.dir/build: CMakeFiles/shadow_base.dir/cmake_device_link.o

.PHONY : CMakeFiles/shadow_base.dir/build

# Object files for target shadow_base
shadow_base_OBJECTS = \
"CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.o" \
"CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.o" \
"CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.o" \
"CMakeFiles/shadow_base.dir/Render_DS/Image.cu.o" \
"CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.o" \
"CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.o" \
"CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.o" \
"CMakeFiles/shadow_base.dir/arg_parse.cpp.o" \
"CMakeFiles/shadow_base.dir/main.cu.o"

# External object files for target shadow_base
shadow_base_EXTERNAL_OBJECTS =

shadow_base: CMakeFiles/shadow_base.dir/Data_Renderer/data_rasterizer.cu.o
shadow_base: CMakeFiles/shadow_base.dir/Data_Renderer/ray_intersect.cu.o
shadow_base: CMakeFiles/shadow_base.dir/Data_Renderer/renderer.cu.o
shadow_base: CMakeFiles/shadow_base.dir/Render_DS/Image.cu.o
shadow_base: CMakeFiles/shadow_base.dir/Render_DS/mesh.cpp.o
shadow_base: CMakeFiles/shadow_base.dir/Render_DS/model_loader.cpp.o
shadow_base: CMakeFiles/shadow_base.dir/Render_DS/ppc.cpp.o
shadow_base: CMakeFiles/shadow_base.dir/arg_parse.cpp.o
shadow_base: CMakeFiles/shadow_base.dir/main.cu.o
shadow_base: CMakeFiles/shadow_base.dir/build.make
shadow_base: Dep/fmt/libfmtd.a
shadow_base: Dep/tinyobjloader/libtinyobjloader.a
shadow_base: CMakeFiles/shadow_base.dir/cmake_device_link.o
shadow_base: CMakeFiles/shadow_base.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable shadow_base"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/shadow_base.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/shadow_base.dir/build: shadow_base

.PHONY : CMakeFiles/shadow_base.dir/build

CMakeFiles/shadow_base.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/shadow_base.dir/cmake_clean.cmake
.PHONY : CMakeFiles/shadow_base.dir/clean

CMakeFiles/shadow_base.dir/depend:
	cd /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build /home/sheng30/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles/shadow_base.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/shadow_base.dir/depend

