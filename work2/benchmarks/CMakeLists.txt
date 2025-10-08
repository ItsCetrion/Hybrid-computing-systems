cmake_minimum_required(VERSION 3.10)
project(work2_benchmarks LANGUAGES CXX CUDA)

include(../cmake/deps/gbench.cmake)
include(../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work1_benchmarks ${SRC})

target_compile_features(work2_benchmarks PRIVATE cxx_std_20)
target_compile_options(work2_benchmarks PRIVATE -Wall -O3)

target_include_directories(work2_benchmarks PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${benchmark_SOURCE_DIR}/include
    ${benchmark_BINARY_DIR}/include
)

target_link_libraries(work2_benchmarks PRIVATE
    work2::core
    CUDA::cudart
    benchmark::benchmark
    Eigen3::Eigen
)

set_target_properties(work2_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)