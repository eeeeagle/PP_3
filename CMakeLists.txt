cmake_minimum_required (VERSION 3.8)

project(PP_3)

set(APP_ICON_RESOURCE_WINDOWS "${CMAKE_CURRENT_SOURCE_DIR}/appicon.rc")
add_executable (PP_3 "src/Source.cpp" "src/Matrix.hpp" ${APP_ICON_RESOURCE_WINDOWS})

find_package(MPI REQUIRED)
target_link_libraries(PP_3 PUBLIC MPI::MPI_CXX)


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET PP_3 PROPERTY CXX_STANDARD 20)
endif()
