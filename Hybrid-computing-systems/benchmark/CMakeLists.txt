cmake_minimum_required(VERSION 3.14)
project(BenchmarkExample)

# Настройка стандарта C++ для всего проекта
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Опция для включения бенчмарков
option(BUILD_BENCHMARKS "Build benchmark tests" ON)

if(BUILD_BENCHMARKS)
    # Загрузка Google Benchmark
    include(FetchContent)
    
    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.0.zip
    )
    
    FetchContent_MakeAvailable(googlebenchmark)
    
    # Добавить бенчмарки
    add_executable(Hybrid-computing-systems
        benchmarks/src/vector_benchmark.cpp
    )
    
    target_link_libraries(benchmarks 
        benchmark::benchmark
        # другие ваши библиотеки
    )
    
    # Добавить include директории
    target_include_directories(benchmarks PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()