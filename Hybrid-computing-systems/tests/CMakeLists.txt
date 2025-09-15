cmake_minimum_required(VERSION 3.10)
project(work1_tests LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work1_tests ${SRC})

target_compile_features(work1_tests PRIVATE cxx_std_20)
target_compile_options(work1_tests PRIVATE -Wall -O3)

target_include_directories(work1_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/Core/include
    ${gtest_SOURCE_DIR}/include
    ${gtest_BINARY_DIR}/include
)

target_link_libraries(work1_tests PRIVATE
    work1::core
    CUDA::cudart
    GTest::gtest_main
)

enable_testing()
add_test(NAME work1_tests COMMAND work1_tests)

set_target_properties(work1_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)