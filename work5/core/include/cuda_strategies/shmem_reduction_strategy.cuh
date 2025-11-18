#pragma once

#include "vector_accessor.cuh"
#include "../utils/cuda_utils/cuda_utils.cuh"
#include "device_memory_block.cuh"
#include "../kernels/kernel_vecred_nobr.cuh"


template <class T>
class ShmemReductionStrategy {
    public:
        void static sum(const VectorAccessor<T> vector, T *result) {

            constexpr std::size_t blockSize = 256;
            constexpr std::size_t unrollFactor = 4;
            std::size_t gridSize = cuda_utils::calcGridSize(vector.size(), blockSize, unrollFactor);
            std::size_t shmemSize = blockSize * sizeof(T);

            kernel_vecred_nobr<T, unrollFactor><<<gridSize, blockSize, shmemSize>>>(vector, result);

        }
};