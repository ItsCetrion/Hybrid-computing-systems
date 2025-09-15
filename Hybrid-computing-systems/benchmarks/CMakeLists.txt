cmake_minimum_required(VERSION 3.10)
project(work1_benchmarks LANGUAGES CXX)

# ВАЖНО: Сначала подключаем benchmark
include(FetchContent)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.2
)
FetchContent_MakeAvailable(benchmark)

# Затем остальные зависимости
find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(work1_benchmarks ${SRC})

target_compile_features(work1_benchmarks PRIVATE cxx_std_20)
target_compile_options(work1_benchmarks PRIVATE -Wall -O3)

# ВАЖНО: Правильные include директории
target_include_directories(work1_benchmarks PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${benchmark_SOURCE_DIR}/include
    ${benchmark_BINARY_DIR}/include
)

target_link_libraries(work1_benchmarks PRIVATE
    work1::core
    CUDA::cudart
    benchmark::benchmark
)

set_target_properties(work1_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)