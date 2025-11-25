#include <benchmark/benchmark.h>

#include <cuda_timer.hpp>
#include "vector.cuh"
#include "device_memory_block.cuh"
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
        CUDATimer timer(elapsed_time);
        std::size_t gridSize = cuda_utils::calcGridSize(vector.size(), blockSize, unrollFactor);
        std::size_t shmemSize = blockSize * sizeof(float);
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
  constexpr std::size_t warpSize = 32;
  constexpr std::size_t blockSize = 256;
  constexpr std::size_t unrollFactor = 4;
  for (auto _ : state)
  {
    float elapsed_time = 0; 
    {
        CUDATimer timer(elapsed_time);
        std::size_t gridSize = cuda_utils::calcGridSize(vector.size(), blockSize, unrollFactor);
        std::size_t numWarps = (blockSize + warpSize - 1) / warpSize;
        std::size_t shmemSize = numWarps * sizeof(float);
        kernel_vecred_br<float, unrollFactor><<<gridSize, blockSize, shmemSize>>>(vector.accessor(), result.data());
    }

    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();
    
    state.SetIterationTime(elapsed_time);
  }
}

void* operator new(std::size_t bytes);  // Dumb clangd!

constexpr int multiplier = 2;
constexpr std::uint64_t min_n = 1 << 3;
constexpr std::uint64_t max_n = 1ull << 31;
constexpr auto unit = benchmark::kMillisecond;

BENCHMARK(BM_CUDAShmemReductionSumGPU)
    ->Name("CUDA Shmem Reduction Sum (GPU)")
    ->RangeMultiplier(multiplier)
    ->Range(min_n, max_n)
    ->Unit(unit)
    ->Iterations(1000)
    ->UseManualTime();

BENCHMARK(BM_CUDAWarmShuffleReductionSumGPU)
    ->Name("CUDA Warp Shuffle Reduction Sum (GPU)")
    ->RangeMultiplier(multiplier)
    ->Range(min_n, max_n)
    ->Unit(unit)
    ->Iterations(1000)
    ->UseManualTime();

BENCHMARK_MAIN();  // NOLINT