cmake_minimum_required(VERSION 3.10)
project(work2_core LANGUAGES CUDA)

include(../cmake/configs/cuda_arch.cmake)

find_package(CUDAToolkit REQUIRED)

add_library(work2::core ALIAS work2_core)

set_target_properties(work1_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/core
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/core
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/core
)

set_target_minimum_cuda_arch(work2_core 75)

target_compile_features(work2_core PRIVATE cuda_std_20)

target_compile_options(work2_core PRIVATE -Wall -O3 -Wno-deprecated-gpu-targets)

target_include_directories(work2_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(work1_core PROPERTIES CUDA_ARCHITECTURES "75")