cmake_minimum_required(VERSION 3.10)
project(work2_core LANGUAGES CUDA)

include(../cmake/configs/cuda_arch.cmake)

find_package(CUDAToolkit REQUIRED)

add_library(work2_core INTERFACE)
add_library(work2::core ALIAS work2_core)

target_include_directories(work2_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(work2_core INTERFACE cuda_std_20)

# Для INTERFACE библиотеки используем INTERFACE компиляционные опции
target_compile_options(work2_core INTERFACE 
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -O3>
)

# Установка архитектуры CUDA (адаптируйте функцию для INTERFACE)
set_target_minimum_cuda_arch(work2_core 75)