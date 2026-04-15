cmake_minimum_required(VERSION 3.22)

project(fusion_v4l2_deepstream_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-video-1.0 glib-2.0 gobject-2.0)

set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream-7.1" CACHE PATH "DeepStream root directory")
set(DEEPSTREAM_INCLUDE_DIR "${DEEPSTREAM_ROOT}/sources/includes" CACHE PATH "DeepStream include directory")
set(DEEPSTREAM_LIB_DIR "${DEEPSTREAM_ROOT}/lib" CACHE PATH "DeepStream library directory")

find_library(NVDS_META_LIB nvds_meta HINTS "${DEEPSTREAM_LIB_DIR}")
find_library(NVDSGST_META_LIB nvdsgst_meta HINTS "${DEEPSTREAM_LIB_DIR}")

foreach(required_lib NVDS_META_LIB NVDSGST_META_LIB)
    if(NOT ${required_lib})
        message(FATAL_ERROR "Required DeepStream library not found: ${required_lib}")
    endif()
endforeach()

add_executable(fusion_v4l2_deepstream_app
    src/main.cpp)

target_include_directories(fusion_v4l2_deepstream_app PRIVATE
    "${DEEPSTREAM_INCLUDE_DIR}"
    ${GST_INCLUDE_DIRS})

target_compile_options(fusion_v4l2_deepstream_app PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    ${GST_CFLAGS_OTHER})

target_link_directories(fusion_v4l2_deepstream_app PRIVATE
    "${DEEPSTREAM_LIB_DIR}"
    ${GST_LIBRARY_DIRS})

target_link_libraries(fusion_v4l2_deepstream_app PRIVATE
    ${GST_LIBRARIES}
    ${NVDS_META_LIB}
    ${NVDSGST_META_LIB})

install(TARGETS fusion_v4l2_deepstream_app RUNTIME DESTINATION bin)
