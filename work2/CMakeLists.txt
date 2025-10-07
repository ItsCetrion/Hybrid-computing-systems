cmake_minimum_required(VERSION 3.10)
project(work1 LANGUAGES CXX CUDA)

option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_BENCH "Enable benchmarks" OFF)

# Исправляем имена папок согласно вашей структуре
add_subdirectory(core)

if(ENABLE_TESTS)
    add_subdirectory(Tests)
endif()

if(ENABLE_BENCH)
    add_subdirectory(benchmarks)
endif()