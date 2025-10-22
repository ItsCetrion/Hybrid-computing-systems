cmake_minimum_required(VERSION 3.10)
project(work3_tests LANGUAGES CUDA)

include(../cmake/deps/gtest.cmake)
include(../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/native_matrix_multiply_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/shmem_matrix_multiply_test.cu
)

add_executable(work3_tests ${TEST_SOURCES})

set_target_properties(work3_tests PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

target_compile_features(work3_tests PRIVATE 
    cuda_std_20
)

target_compile_options(work3_tests PRIVATE 
    -O3 
    -Wno-deprecated-gpu-targets
    --expt-relaxed-constexpr  # Разрешает constexpr в device code
)

target_include_directories(work3_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${gtest_SOURCE_DIR}/include
    ${gtest_BINARY_DIR}/include
)

target_link_libraries(work3_tests PRIVATE
    work3::core
    CUDA::cudart
    GTest::gtest_main
    Eigen3::Eigen
)

enable_testing()
add_test(NAME work3_tests COMMAND work3_tests)

set_target_properties(work3_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)