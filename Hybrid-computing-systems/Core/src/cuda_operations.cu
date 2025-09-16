#include <cuda_operations.h>
#include <cuda_runtime.h>


// Ядро вычислений
__global__ void addVectorsKernel(const float *d_a, const float *d_b, float *d_result, const int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        d_result[idx] = d_a[idx] + d_b[idx];
    }
}

void addVectorsExtended(const float *h_a, const float *h_b, float *h_result, const int n, const int blockSize, const int gridSize) {
    std::size_t array_size = n * sizeof(float);
    float *d_a, *d_b, *d_result;

    cudaMalloc(&d_a, array_size);
    cudaMalloc(&d_b, array_size);
    cudaMalloc(&d_result, array_size);

    cudaMemcpy(d_a, h_a, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, array_size, cudaMemcpyHostToDevice);

    addVectorsKernel<<<gridSize, blockSize>>>(d_a, d_b, d_result, n);

    cudaMemcpy(h_result, d_result, array_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
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
    cudaMalloc(&ptr, size);
    return ptr;
}

void cuda_free(void *ptr) {
    cudaFree(ptr);
}


void cuda_memcpy_host_to_device(void *dst, const void *src, std::size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}


void cuda_memcpy_device_to_host(void *dst, const void *src, std::size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

