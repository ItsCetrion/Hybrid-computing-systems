#include <kernel_matrix_multiply.cuh>
#include <matrix.cuh>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_timer.hpp>
#include <cudagh.hpp>


static void BM_EigenMatrixAddCPU(benchmark::State& state)
{
  auto size = state.range(0);

  Eigen::MatrixXf a = Eigen::MatrixXf(size, size);
  Eigen::MatrixXf b = Eigen::MatrixXf(size, size);
  Eigen::MatrixXf result(size, size);

  for (auto _ : state)
  {
    result = a * b;
    benchmark::DoNotOptimize(result.data());
    benchmark::ClobberMemory();
  }
}

static void BM_CUDAMatrixAddGPU(benchmark::State& state)
{
  auto size = state.range(0);

  Matrix<float> a(size, size);
  Matrix<float> b(size, size);
  Matrix<float> c(size, size);

  for (auto _ : state)
  {
    float elapsed_time = 0;
    {
      CUDATimer timer(elapsed_time);
      kernel_matmul_naive<<<cudagh::cover(size, 128), 128>>>(
        a.getAccessor(), b.getAccessor(), c.getAccessor());
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      std::cerr << "CUDA launch error: " << cudaGetErrorString(err) << std::endl;
    benchmark::DoNotOptimize(elapsed_time);
    benchmark::ClobberMemory();

    state.SetIterationTime(elapsed_time);
  }
}

void* operator new(std::size_t bytes);  // Dumb clangd!

constexpr int multiplier = 2;
constexpr auto range = std::make_pair(16, 1024);
constexpr auto unit = benchmark::kMillisecond;

BENCHMARK(BM_EigenMatrixAddCPU)
    ->Name("Eigen Matrix Multiplication (CPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseRealTime()
    ->MeasureProcessCPUTime();

BENCHMARK(BM_CUDAMatrixAddGPU)
    ->Name("CUDA Matrix Multiplication (GPU)")
    ->RangeMultiplier(multiplier)
    ->Ranges({range})
    ->Unit(unit)
    ->UseManualTime();

BENCHMARK_MAIN();  // NOLINT