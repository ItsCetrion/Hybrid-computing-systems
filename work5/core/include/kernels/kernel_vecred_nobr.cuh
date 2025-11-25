#pragma once

#include <cuda_runtime.h>

#include "vector_accessor.cuh"


template <class T, std::size_t unrollFactor>
__global__ void kernel_vecred_nobr(const VectorAccessor<T> vector, T *result) {

    extern __shared__ T partialSum[];

    std::size_t tid = threadIdx.x;
    std::size_t i = blockIdx.x * (unrollFactor * blockDim.x) + tid;

    T sum = 0;

    #pragma unroll
    for (std::size_t u = 0; u < unrollFactor; ++u) {
        std::size_t index = i + u * blockDim.x;
        if (index < vector.size()) {
            sum += vector[index];
        }
    }
    partialSum[tid] = sum;

    __syncthreads();

    for (std::size_t s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            partialSum[tid] += partialSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, partialSum[0]);
    }

}