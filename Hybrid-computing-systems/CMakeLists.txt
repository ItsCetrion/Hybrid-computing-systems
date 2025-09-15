cmake_minimum_required(VERSION 3.10)  # Понижаем версию для совместимости
project(work1 LANGUAGES CXX CUDA)

option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_BENCH "Enable benchmarks" ON)

# Исправляем имена папок согласно вашей структуре
add_subdirectory(Core)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

if(ENABLE_BENCH)
    add_subdirectory(benchmarks)
endif()