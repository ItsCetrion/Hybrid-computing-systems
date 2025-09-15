#include <benchmark/benchmark.h>  // Правильный include
#include "Core.h"                 // Правильный include

// Простой бенчмарк для примера
static void BM_Empty(benchmark::State& state) {  // Правильно: benchmark::State&
    for (auto _ : state) {
        // Ваш код для бенчмарка
        benchmark::DoNotOptimize(_);
    }
}
BENCHMARK(BM_Empty);

// Бенчмарк, использующий вашу Core библиотеку
static void BM_CoreFunction(benchmark::State& state) {
    for (auto _ : state) {
        // Пример вызова функции из Core
        // int result = core_function(state.range(0));
        // benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_CoreFunction)->Arg(100)->Arg(1000);

BENCHMARK_MAIN();