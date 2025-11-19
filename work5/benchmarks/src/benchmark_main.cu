#include <benchmark/benchmark.h>

#include <cuda_timer.hpp>
#include "kernels/kernel_vecred_br.cuh"
#include "kernels/kernel_vecred_nobr.cuh"


static void BM_CUDAShmemReductionSumGPU(benchmark::State& state)
{
  auto size = state.range(0);
  Vector<float> vector(size);
  DeviceMemoryBlock<float> result(1);
  float hostSum = 0;
  result.copyFromHost(&hostSum);
  constexpr std::size_t blockSize = 256;
  constexpr std::size_t unrollFactor = 4;
  for (auto _ : state)
  {
    float elapsed_time = 0; 
    {
        std::size_t gridSize = cuda_utils::calcGridSize(vector.size(), blockSize, unrollFactor);
        std::size_t shmemSize = blockSize * sizeof(T);
        kernel_vecred_nobr<float, unrollFactor><<<gridSize, blockSize, shmemSize>>>(vector.accessor(), result.data());
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();
    
    state.SetIterationTime(elapsed_time);
  }
}

static void BM_CUDAWarmShuffleReductionSumGPU(benchmark::State& state)
{
  auto size = state.range(0);
  Vector<float> vector(size);
  DeviceMemoryBlock<float> result(1);
  float hostSum = 0;
  result.copyFromHost(&hostSum);
  constexpr std::size_t blockSize = 256;
  constexpr std::size_t unrollFactor = 4;
  for (auto _ : state)
  {
    float elapsed_time = 0; 
    {
        std::size_t gridSize = cuda_utils::calcGridSize(vector.size(), blockSize, unrollFactor);
        std::size_t shmemSize = blockSize * sizeof(T);
        kernel_vecred_br<float, unrollFactor><<<gridSize, blockSize, shmemSize>>>(vector.accessor(), result.data());
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();
    
    state.SetIterationTime(elapsed_time);
  }
}

void* operator new(std::size_t bytes);  // Dumb clangd!

constexpr int multiplier = 2;
constexpr auto range = std::make_pair(8, 1<<31);
constexpr auto unit = benchmark::kMillisecond;

BENCHMARK(BM_CUDAShmemMatrixAddGPU)
    ->Name("CUDA Shmem Reduction Sum (GPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK(BM_CUDAWmmaMatrixAddGPU)
    ->Name("CUDA Warp Shuffle Reduction Sum (GPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();  // NOLINT