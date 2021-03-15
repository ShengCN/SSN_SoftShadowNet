# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /opt/cmake-3.17.1-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.17.1-Linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build

# Include any dependencies generated for this target.
include CMakeFiles/hard_shadow_fov.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hard_shadow_fov.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hard_shadow_fov.dir/flags.make

CMakeFiles/hard_shadow_fov.dir/force_fov.cu.o: CMakeFiles/hard_shadow_fov.dir/flags.make
CMakeFiles/hard_shadow_fov.dir/force_fov.cu.o: ../force_fov.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/hard_shadow_fov.dir/force_fov.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/force_fov.cu -o CMakeFiles/hard_shadow_fov.dir/force_fov.cu.o

CMakeFiles/hard_shadow_fov.dir/force_fov.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/hard_shadow_fov.dir/force_fov.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/hard_shadow_fov.dir/force_fov.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/hard_shadow_fov.dir/force_fov.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/hard_shadow_fov.dir/common.cpp.o: CMakeFiles/hard_shadow_fov.dir/flags.make
CMakeFiles/hard_shadow_fov.dir/common.cpp.o: ../common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/hard_shadow_fov.dir/common.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hard_shadow_fov.dir/common.cpp.o -c /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/common.cpp

CMakeFiles/hard_shadow_fov.dir/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hard_shadow_fov.dir/common.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/common.cpp > CMakeFiles/hard_shadow_fov.dir/common.cpp.i

CMakeFiles/hard_shadow_fov.dir/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hard_shadow_fov.dir/common.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/common.cpp -o CMakeFiles/hard_shadow_fov.dir/common.cpp.s

CMakeFiles/hard_shadow_fov.dir/mesh.cpp.o: CMakeFiles/hard_shadow_fov.dir/flags.make
CMakeFiles/hard_shadow_fov.dir/mesh.cpp.o: ../mesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/hard_shadow_fov.dir/mesh.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hard_shadow_fov.dir/mesh.cpp.o -c /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/mesh.cpp

CMakeFiles/hard_shadow_fov.dir/mesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hard_shadow_fov.dir/mesh.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/mesh.cpp > CMakeFiles/hard_shadow_fov.dir/mesh.cpp.i

CMakeFiles/hard_shadow_fov.dir/mesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hard_shadow_fov.dir/mesh.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/mesh.cpp -o CMakeFiles/hard_shadow_fov.dir/mesh.cpp.s

CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.o: CMakeFiles/hard_shadow_fov.dir/flags.make
CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.o: ../model_loader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.o -c /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/model_loader.cpp

CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/model_loader.cpp > CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.i

CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/model_loader.cpp -o CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.s

CMakeFiles/hard_shadow_fov.dir/ppc.cpp.o: CMakeFiles/hard_shadow_fov.dir/flags.make
CMakeFiles/hard_shadow_fov.dir/ppc.cpp.o: ../ppc.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/hard_shadow_fov.dir/ppc.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hard_shadow_fov.dir/ppc.cpp.o -c /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/ppc.cpp

CMakeFiles/hard_shadow_fov.dir/ppc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hard_shadow_fov.dir/ppc.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/ppc.cpp > CMakeFiles/hard_shadow_fov.dir/ppc.cpp.i

CMakeFiles/hard_shadow_fov.dir/ppc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hard_shadow_fov.dir/ppc.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/ppc.cpp -o CMakeFiles/hard_shadow_fov.dir/ppc.cpp.s

# Object files for target hard_shadow_fov
hard_shadow_fov_OBJECTS = \
"CMakeFiles/hard_shadow_fov.dir/force_fov.cu.o" \
"CMakeFiles/hard_shadow_fov.dir/common.cpp.o" \
"CMakeFiles/hard_shadow_fov.dir/mesh.cpp.o" \
"CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.o" \
"CMakeFiles/hard_shadow_fov.dir/ppc.cpp.o"

# External object files for target hard_shadow_fov
hard_shadow_fov_EXTERNAL_OBJECTS =

CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o: CMakeFiles/hard_shadow_fov.dir/force_fov.cu.o
CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o: CMakeFiles/hard_shadow_fov.dir/common.cpp.o
CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o: CMakeFiles/hard_shadow_fov.dir/mesh.cpp.o
CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o: CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.o
CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o: CMakeFiles/hard_shadow_fov.dir/ppc.cpp.o
CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o: CMakeFiles/hard_shadow_fov.dir/build.make
CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o: libtinyobj.a
CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o: CMakeFiles/hard_shadow_fov.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CUDA device code CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hard_shadow_fov.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hard_shadow_fov.dir/build: CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o

.PHONY : CMakeFiles/hard_shadow_fov.dir/build

# Object files for target hard_shadow_fov
hard_shadow_fov_OBJECTS = \
"CMakeFiles/hard_shadow_fov.dir/force_fov.cu.o" \
"CMakeFiles/hard_shadow_fov.dir/common.cpp.o" \
"CMakeFiles/hard_shadow_fov.dir/mesh.cpp.o" \
"CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.o" \
"CMakeFiles/hard_shadow_fov.dir/ppc.cpp.o"

# External object files for target hard_shadow_fov
hard_shadow_fov_EXTERNAL_OBJECTS =

hard_shadow_fov: CMakeFiles/hard_shadow_fov.dir/force_fov.cu.o
hard_shadow_fov: CMakeFiles/hard_shadow_fov.dir/common.cpp.o
hard_shadow_fov: CMakeFiles/hard_shadow_fov.dir/mesh.cpp.o
hard_shadow_fov: CMakeFiles/hard_shadow_fov.dir/model_loader.cpp.o
hard_shadow_fov: CMakeFiles/hard_shadow_fov.dir/ppc.cpp.o
hard_shadow_fov: CMakeFiles/hard_shadow_fov.dir/build.make
hard_shadow_fov: libtinyobj.a
hard_shadow_fov: CMakeFiles/hard_shadow_fov.dir/cmake_device_link.o
hard_shadow_fov: CMakeFiles/hard_shadow_fov.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable hard_shadow_fov"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hard_shadow_fov.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hard_shadow_fov.dir/build: hard_shadow_fov

.PHONY : CMakeFiles/hard_shadow_fov.dir/build

CMakeFiles/hard_shadow_fov.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hard_shadow_fov.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hard_shadow_fov.dir/clean

CMakeFiles/hard_shadow_fov.dir/depend:
	cd /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build /home/ysheng/Documents/cpu_hard_shadow_renderer/cpu_hard_shadow_renderer_src/build/CMakeFiles/hard_shadow_fov.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hard_shadow_fov.dir/depend

