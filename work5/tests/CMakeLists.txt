cmake_minimum_required(VERSION 3.10)
project(work5_tests LANGUAGES CUDA)

include(../cmake/deps/gtest.cmake)
include(../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

set(TEST_SOURCES
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/matrix_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ShmemReduction_sum_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/WarpShuffleReduction_sum_test.cu
)

add_executable(work5_tests ${TEST_SOURCES})

set_target_properties(work5_tests PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75"
)

target_compile_features(work5_tests PRIVATE 
    cuda_std_20
)

target_compile_options(work5_tests PRIVATE 
    -O3 
    -Wno-deprecated-gpu-targets
    --expt-relaxed-constexpr  # Разрешает constexpr в device code
)

target_include_directories(work5_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${gtest_SOURCE_DIR}/include
    ${gtest_BINARY_DIR}/include
)

target_link_libraries(work5_tests PRIVATE
    work5::core
    CUDA::cudart
    GTest::gtest_main
    Eigen3::Eigen
)

enable_testing()
add_test(NAME work5_tests COMMAND work5_tests)

set_target_properties(work5_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)