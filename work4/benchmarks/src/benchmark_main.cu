#include <benchmark/benchmark.h>

#include <cuda_timer.hpp>
// #include "kernels/kernel_matrix_multiply.cuh"
#include "kernels/kernel_matmul_shmem.cuh"
#include "kernels/kernel_matmul_wmma.cuh"
#include <matrix.cuh>
// #include "cuda_strategies/naive_matmul_strategy.cuh"
#include "cuda_strategies/shmem_matmul_strategy.cuh"
#include "cuda_strategies/wmma_matmul_strategy.cuh"


static void BM_CUDAShmemMatrixAddGPU(benchmark::State& state)
{
  auto size = state.range(0);

  Matrix<float> a(size, size);
  Matrix<float> b(size, size);
  Matrix<float> c(size, size);

  constexpr std::size_t blockSize = 16;

  for (auto _ : state)
  {
    float elapsed_time = 0;
    {
      CUDATimer timer(elapsed_time);
      auto [blocks, threads] = cuda_utils::calcGridSize(blockSize, size, size);
      kernel_matmul_shmem<float, blockSize><<<blocks, threads>>>(
        a.getAccessor(), b.getAccessor(), c.getAccessor());
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }
}

static void BM_CUDAWmmaMatrixAddGPU(benchmark::State& state)
{
  auto size = state.range(0);

  Matrix<half> a(size, size);
  Matrix<half> b(size, size);
  Matrix<float> c(size, size);

  for (auto _ : state)
  {
    float elapsed_time = 0;
    {
      CUDATimer timer(elapsed_time);
      auto [gridDim, blockDim] = cuda_utils::calcGridSizeWMMA(size, size);
      kernel_matmul_wmma<<<gridDim, blockDim>>>(
        a.getAccessor(), b.getAccessor(), c.getAccessor());
    }
    
    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }
}

void* operator new(std::size_t bytes);  // Dumb clangd!

constexpr int multiplier = 2;
constexpr auto range = std::make_pair(16, 1024);
constexpr auto unit = benchmark::kMillisecond;

BENCHMARK(BM_CUDAShmemMatrixAddGPU)
    ->Name("CUDA Shmem Matrix Multiplication (GPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_CUDAWmmaMatrixAddGPU)
    ->Name("CUDA WMMA Matrix Multiplication (GPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();  // NOLINT