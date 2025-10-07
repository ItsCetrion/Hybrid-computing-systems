cmake_minimum_required(VERSION 3.10)
project(work2_tests LANGUAGES CUDA)

include(../cmake/deps/gtest.cmake)
include(../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_multiply_test.cu
)

add_executable(work2_tests ${TEST_SOURCES})

set_target_properties(work2_tests PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

target_compile_features(work2_tests PRIVATE 
    cuda_std_20
)

target_compile_options(work2_tests PRIVATE 
    -O3 
    -Wno-deprecated-gpu-targets
    --expt-relaxed-constexpr  # Разрешает constexpr в device code
)

target_include_directories(work2_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${gtest_SOURCE_DIR}/include
    ${gtest_BINARY_DIR}/include
)

target_link_libraries(work2_tests PRIVATE
    work2::core
    CUDA::cudart
    GTest::gtest_main
    Eigen3::Eigen
)

enable_testing()
add_test(NAME work2_tests COMMAND work2_tests)

set_target_properties(work2_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)