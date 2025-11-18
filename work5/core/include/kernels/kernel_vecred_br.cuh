#pragma once

#include <cuda_runtime.h>

#include "vector_accessor.cuh"


template <class T>
__device__ __forceinline__ T warpReduceSum(T sum) {
    constexpr unsigned int fullMask = 0xffffffffU;
    for (std::size_t offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(fullMask, sum, offset);
    }
    return sum;
}

template <class T, std::size_t unrollFactor>
__global__ void kernel_vecred_br(const VectorAccessor<T> vector, T *result) {

    extern __shared__ T shmem[];

    std::size_t tid = threadIdx.x;
    std::size_t i = blockIdx.x * (unrollFactor * blockDim.x) + tid;
    std::size_t warpId = tid / warpSize;
    std::size_t laneId = tid % warpSize;

    T sum = 0;

    #pragma unroll
    for (std::size_t u = 0; u < unrollFactor; ++u) {
        std::size_t index = i + u * blockDim.x;
        if (index < vector.size()) {
            sum += vector[index];
        }
    }

    sum = warpReduceSum(sum);

    if (laneId == 0) {
        shmem[warpId] = sum;
    }

    __syncthreads();

    if (warpId == 0) {
        sum = (tid < (blockDim.x + warpSize - 1) / warpSize) ? shmem[laneId] : 0;
        sum = warpReduceSum(sum);

        if (laneId == 0) {
            atomicAdd(result, sum);
        }
    }

}