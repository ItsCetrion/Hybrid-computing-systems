cmake_minimum_required(VERSION 3.10)
project(work4 LANGUAGES CXX CUDA)

option(ENABLE_TESTS "Enable tests" OFF)
option(ENABLE_BENCH "Enable benchmarks" ON)

add_subdirectory(core)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(ENABLE_BENCH)
    add_subdirectory(benchmarks)
endif()