#include <cuda_operations.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>


// Ядро вычислений
__global__ void addVectorsKernel(const float *d_a, const float *d_b, float *d_result, const int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        d_result[idx] = d_a[idx] + d_b[idx];
    }
}

void addVectorsExtended(const float *h_a, const float *h_b, float *h_result, const int n, const int blockSize, const int gridSize) {
    std::size_t array_size = n * sizeof(float);

    float *d_a = static_cast<float*>(cuda_malloc(array_size));
    float *d_b = static_cast<float*>(cuda_malloc(array_size));
    float *d_result = static_cast<float*>(cuda_malloc(array_size));

    cuda_memcpy_host_to_device(d_a, h_a, array_size);
    cuda_memcpy_host_to_device(d_b, h_b, array_size);

    addVectorsKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, n);

    cuda_memcpy_device_to_host(h_result, d_result, array_size);

    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_result);
}

void addVectorsOptimal(const float *h_a, const float *h_b, float *h_result, const int n) {
    int blockSize = 128;
    int gridSize = (n + blockSize - 1) / blockSize;
    addVectorsExtended(h_a, h_b, h_result, n, blockSize, gridSize);
}

void addVectorsKernelRun(const float *d_a, const float *d_b, float *d_result, const int n, const int blockSize, const int gridSize, float *kernel_ms) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    addVectorsKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(kernel_ms, start, stop);
}

void* cuda_malloc(std::size_t size) {
    void *ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMalloc failed"); 
    }
    return ptr;
}

void cuda_free(void *ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaFree failed"); 
    }
}

void cuda_memcpy_host_to_device(void *dst, const void *src, std::size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy HtoD failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemcpy failed"); 
    }
}

void cuda_memcpy_device_to_host(void *dst, const void *src, std::size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy DtoH failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemcpy failed"); 
    }
}

