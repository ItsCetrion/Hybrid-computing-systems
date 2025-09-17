#include <benchmark/benchmark.h>
#include <cuda_operations.h>
#include <CPU_operations.h>


static void BM_VectorsAddCPU(benchmark::State& state) {
    int n = state.range(0);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_result = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    for (auto _ : state) {
        addVectorsCPU(h_a, h_b, h_result, n);
        benchmark::DoNotOptimize(h_result);
    }
    delete[] h_a;
    delete[] h_b;
    delete[] h_result;
}


static void BM_VectorsAddGPUCore(benchmark::State& state) {
    int n = state.range(0);
    std::size_t array_size = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_result = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    float *d_a = static_cast<float*>(cuda_malloc(array_size));
    float *d_b = static_cast<float*>(cuda_malloc(array_size));
    float *d_result = static_cast<float*>(cuda_malloc(array_size));

    cuda_memcpy_host_to_device(d_a, h_a, array_size);
    cuda_memcpy_host_to_device(d_b, h_b, array_size);

    float *kernel_ms = new float;

    int blockSize = 128;
    int gridSize = (n + blockSize - 1) / blockSize;

    for (auto _ : state) {
        addVectorsKernelRun(d_a, d_b, d_result, n, blockSize, gridSize, kernel_ms);
        state.SetIterationTime(*kernel_ms);
        benchmark::DoNotOptimize(h_result);
    }

    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_result);
    delete kernel_ms;
    delete[] h_a;
    delete[] h_b;
    delete[] h_result;
}


static void BM_VectorsAddGPUFull(benchmark::State& state) {
    int n = state.range(0);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_result = new float[n];

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    for (auto _ : state) {
        addVectorsOptimal(h_a, h_b, h_result, n);
        benchmark::DoNotOptimize(h_result);
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_result;
}


BENCHMARK(BM_VectorsAddGPUFull)
    ->Name("VectorsAddGPUFull")
    ->RangeMultiplier(2)
    ->Range(1<<10, 1<<22);
    
BENCHMARK(BM_VectorsAddGPUCore)
    ->Name("VectorsAddGPUCore")
    ->RangeMultiplier(2)
    ->Range(1<<10, 1<<22);

BENCHMARK(BM_VectorsAddCPU)
    ->Name("VectorAddCPU")
    ->RangeMultiplier(2)
    ->Range(1<<10, 1<<22);


BENCHMARK_MAIN();

