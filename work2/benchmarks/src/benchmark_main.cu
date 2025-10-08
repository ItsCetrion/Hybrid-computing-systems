#include <cudagh.hpp>
#include <kernel_matrix_multiply.cuh>
#include <matrix.cuh>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_timer.hpp> 


static void BM_EigenMatrixAddCPU(benchmark::State& state)
{
  auto n = state.range(0);

  Eigen::MatrixXf a = Eigen::MatrixXf(n, n);
  Eigen::MatrixXf b = Eigen::MatrixXf(n, n);
  Eigen::MatrixXf c = result(n,n);

  for (auto _ : state) {
    result = a * b;
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

void* operator new(std::size_t bytes);  // Dumb clangd!

constexpr int multiplier = 8;
constexpr auto range = std::make_pair(8, 1 << 26);
constexpr auto unit = benchmark::kMillisecond;

BENCHMARK(BM_EigenMatrixAddCPU)
    ->Name("Eigen Vector Addition (CPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

// BENCHMARK(BM_CUDAVectorAddGPU)
//     ->Name("CUDA Vector Addition (GPU)")
//     ->RangeMultiplier(multiplier)
//     ->Ranges({range})
//     ->Unit(unit)
//     ->UseManualTime();

BENCHMARK_MAIN();  // NOLINT