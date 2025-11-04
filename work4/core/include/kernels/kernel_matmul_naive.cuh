#pragma once

#include <cuda_runtime.h>

#include "matrix_accessor.cuh"


template <typename T>
__global__ void kernel_matmul_naive(const MatrixAccessor<T> A, const MatrixAccessor<T> B, MatrixAccessor<T> C) {
    std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < A.nrows() && j < B.ncols()) {
        T sum = 0;
        for (std::size_t k = 0; k < A.ncols(); ++k) {
            sum += A(i, k) * B(k, j);
        }
        C(i, j) = sum;
    }
}