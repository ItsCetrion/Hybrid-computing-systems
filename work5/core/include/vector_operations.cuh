#pragma once

#include "vector_accessor.cuh"
#include "../utils/cuda_utils/cuda_utils.cuh"
#include "vector.cuh"
#include "device_memory_block.cuh"
#include "./cuda_strategies/shmem_reduction_strategy.cuh"
#include "./cuda_strategies/warp_shuffle_reduction_strategy.cuh"


struct ShmemReductionTag {};
struct WarpShuffleTag {};

template <class T>
void sumImpl(const VectorAccessor<T> vector, T *result, ShmemReductionTag) {
    ShmemReductionStrategy<T>::sum(vector, result);
}

template <class T>
void sumImpl(const VectorAccessor<T> vector, T *result, WarpShuffleTag) {
    WarpShuffleReductionStrategy<T>::sum(vector, result);
}

template <class T, class Tag>
T sum(const Vector<T>& vector, Tag tag) {

    DeviceMemoryBlock<T> result(1);
    T hostSum = 0;
    result.copyFromHost(&hostSum);

    sumImpl(vector.accessor(), result.data(), tag);

    cuda_utils::checkCudaKernelErrors();

    result.copyToHost(&hostSum);

    return hostSum;
}