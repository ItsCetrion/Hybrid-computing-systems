cmake_minimum_required(VERSION 3.10)
project(work4_benchmarks LANGUAGES CUDA)

include(../cmake/deps/gbench.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_subdirectory(
     ${CMAKE_CURRENT_SOURCE_DIR}/../utils/cuda_timer
     ${CMAKE_BINARY_DIR}/utils/cuda_timer
 )

add_executable(work4_benchmarks ${SRC})

target_compile_features(work4_benchmarks PRIVATE cuda_std_20)
#target_compile_features(work3_benchmarks PRIVATE cxx_std_20)
target_compile_options(work4_benchmarks PRIVATE -Wall -O3)

target_include_directories(work4_benchmarks PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${PROJECT_SOURCE_DIR}/core/utils/cuda_utils
    ${benchmark_SOURCE_DIR}/include
    ${benchmark_BINARY_DIR}/include
)

target_link_libraries(work4_benchmarks PRIVATE
    work4::core
    benchmark::benchmark
    work4::cuda_timer
)

set_target_properties(work4_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)