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
CMAKE_SOURCE_DIR = /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build

# Include any dependencies generated for this target.
include tinyxml2/CMakeFiles/tinyxml2.dir/depend.make

# Include the progress variables for this target.
include tinyxml2/CMakeFiles/tinyxml2.dir/progress.make

# Include the compile flags for this target's objects.
include tinyxml2/CMakeFiles/tinyxml2.dir/flags.make

tinyxml2/CMakeFiles/tinyxml2.dir/tinyxml2.cpp.o: tinyxml2/CMakeFiles/tinyxml2.dir/flags.make
tinyxml2/CMakeFiles/tinyxml2.dir/tinyxml2.cpp.o: ../tinyxml2/tinyxml2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tinyxml2/CMakeFiles/tinyxml2.dir/tinyxml2.cpp.o"
	cd /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build/tinyxml2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tinyxml2.dir/tinyxml2.cpp.o -c /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/tinyxml2/tinyxml2.cpp

tinyxml2/CMakeFiles/tinyxml2.dir/tinyxml2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tinyxml2.dir/tinyxml2.cpp.i"
	cd /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build/tinyxml2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/tinyxml2/tinyxml2.cpp > CMakeFiles/tinyxml2.dir/tinyxml2.cpp.i

tinyxml2/CMakeFiles/tinyxml2.dir/tinyxml2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tinyxml2.dir/tinyxml2.cpp.s"
	cd /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build/tinyxml2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/tinyxml2/tinyxml2.cpp -o CMakeFiles/tinyxml2.dir/tinyxml2.cpp.s

# Object files for target tinyxml2
tinyxml2_OBJECTS = \
"CMakeFiles/tinyxml2.dir/tinyxml2.cpp.o"

# External object files for target tinyxml2
tinyxml2_EXTERNAL_OBJECTS =

tinyxml2/libtinyxml2d.so.7.1.0: tinyxml2/CMakeFiles/tinyxml2.dir/tinyxml2.cpp.o
tinyxml2/libtinyxml2d.so.7.1.0: tinyxml2/CMakeFiles/tinyxml2.dir/build.make
tinyxml2/libtinyxml2d.so.7.1.0: tinyxml2/CMakeFiles/tinyxml2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libtinyxml2d.so"
	cd /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build/tinyxml2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tinyxml2.dir/link.txt --verbose=$(VERBOSE)
	cd /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build/tinyxml2 && $(CMAKE_COMMAND) -E cmake_symlink_library libtinyxml2d.so.7.1.0 libtinyxml2d.so.7 libtinyxml2d.so

tinyxml2/libtinyxml2d.so.7: tinyxml2/libtinyxml2d.so.7.1.0
	@$(CMAKE_COMMAND) -E touch_nocreate tinyxml2/libtinyxml2d.so.7

tinyxml2/libtinyxml2d.so: tinyxml2/libtinyxml2d.so.7.1.0
	@$(CMAKE_COMMAND) -E touch_nocreate tinyxml2/libtinyxml2d.so

# Rule to build all files generated by this target.
tinyxml2/CMakeFiles/tinyxml2.dir/build: tinyxml2/libtinyxml2d.so

.PHONY : tinyxml2/CMakeFiles/tinyxml2.dir/build

tinyxml2/CMakeFiles/tinyxml2.dir/clean:
	cd /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build/tinyxml2 && $(CMAKE_COMMAND) -P CMakeFiles/tinyxml2.dir/cmake_clean.cmake
.PHONY : tinyxml2/CMakeFiles/tinyxml2.dir/clean

tinyxml2/CMakeFiles/tinyxml2.dir/depend:
	cd /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/tinyxml2 /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build/tinyxml2 /home/ysheng/Documents/Research/SSN_SoftShadowNet/data/src/build/tinyxml2/CMakeFiles/tinyxml2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tinyxml2/CMakeFiles/tinyxml2.dir/depend

