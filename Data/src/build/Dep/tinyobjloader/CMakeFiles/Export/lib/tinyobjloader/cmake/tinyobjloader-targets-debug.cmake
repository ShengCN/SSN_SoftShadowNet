#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tinyobjloader::tinyobjloader" for configuration "Debug"
set_property(TARGET tinyobjloader::tinyobjloader APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(tinyobjloader::tinyobjloader PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libtinyobjloader.so.2.0.0-rc.8"
  IMPORTED_SONAME_DEBUG "libtinyobjloader.so.2"
  )

list(APPEND _IMPORT_CHECK_TARGETS tinyobjloader::tinyobjloader )
list(APPEND _IMPORT_CHECK_FILES_FOR_tinyobjloader::tinyobjloader "${_IMPORT_PREFIX}/lib/libtinyobjloader.so.2.0.0-rc.8" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
