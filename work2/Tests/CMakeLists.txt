cmake_minimum_required(VERSION 3.10)
project(work2_tests LANGUAGES CUDA)

include(../cmake/deps/gtest.cmake)
include(../cmake/deps/eigen.cmake)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work2_tests ${SRC})

target_compile_features(work2_tests PRIVATE 
    cxx_std_20
    cuda_std_20
)

target_compile_options(work2_tests PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Wno-deprecated-gpu-targets>
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
    CUDA_RUNTIME_LIBRARY Shared
)