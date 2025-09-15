cmake_minimum_required(VERSION 3.10)
project(work1_core LANGUAGES CXX CUDA)

include(../cmake/configs/cuda_arch.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_library(work1_core SHARED ${SRC})
add_library(work1::core ALIAS work1_core)

set_target_properties(work1_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/core
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/core
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/core
)

set_target_minimum_cuda_arch(work1_core 75)

target_compile_features(work1_core PRIVATE cuda_std_20)

target_compile_options(work1_core PRIVATE -Wall -O3 -Wno-deprecated-gpu-targets)

target_include_directories(work1_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(work1_core PROPERTIES CUDA_ARCHITECTURES "75")