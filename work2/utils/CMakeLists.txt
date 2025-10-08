cmake_minimum_required(VERSION 3.10)
project(cuda_utils_lib LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Библиотека cuda_timer (реальная, со своими .cpp файлами)
# ============================================================================

file(GLOB_RECURSE CUDA_TIMER_SRC
    cuda_timer/src/*.cpp
)

add_library(cuda_timer STATIC ${CUDA_TIMER_SRC})

target_include_directories(cuda_timer PUBLIC
    cuda_timer/include/
)

target_compile_features(cuda_timer PRIVATE cxx_std_20)
target_link_libraries(cuda_timer PRIVATE CUDA::cudart)

# ============================================================================
# cudagh - просто заголовочная библиотека, НЕ НУЖЕН INTERFACE!
# ============================================================================

add_library(cudagh INTERFACE)
target_include_directories(cudagh INTERFACE cudagh/include/)

target_link_libraries(cuda_utils_lib PRIVATE cuda_timer cudagh)