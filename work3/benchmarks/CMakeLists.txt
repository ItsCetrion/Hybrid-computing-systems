cmake_minimum_required(VERSION 3.10)
project(work3_benchmarks LANGUAGES CUDA)

include(../cmake/deps/gbench.cmake)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/cuda_timer
    ${CMAKE_BINARY_DIR}/utils/cuda_timer
)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/cudagh
    ${CMAKE_BINARY_DIR}/utils/cudagh
)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work3_benchmarks ${SRC})

target_compile_features(work3_benchmarks PRIVATE cuda_std_20)
#target_compile_features(work3_benchmarks PRIVATE cxx_std_20) 
target_compile_options(work3_benchmarks PRIVATE -Wall -O3)

target_include_directories(work3_benchmarks PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${benchmark_SOURCE_DIR}/include
    ${benchmark_BINARY_DIR}/include
)

target_link_libraries(work3_benchmarks PRIVATE
    work3::core
    benchmark::benchmark
    work3::cuda_timer
    work3::cudagh
)

set_target_properties(work3_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)