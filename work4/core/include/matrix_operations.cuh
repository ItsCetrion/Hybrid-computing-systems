#pragma once

#include <cuda_runtime.h>

#include "kernels/kernel_matmul_wmma.cuh"
#include "cuda_strategies/naive_matmul_strategy.cuh"
#include "cuda_strategies/shmem_matmul_strategy.cuh"
#include "cuda_strategies/wmma_matmul_strategy.cuh"
#include "matrix_accessor.cuh"
#include "matrix.cuh"


enum class MatMulStrategy {
    Naive,
    Shmem,
    WMMA
};

template <MatMulStrategy S, class TA, class TB, class TC>
Matrix<TC> multiply(const Matrix<TA>& A, const Matrix<TB>& B) {

    if (A.ncols() != B.nrows()) {
        throw std::runtime_error("Matrices are not compatible for multiplication");
    }
    
    Matrix<TC> result(A.nrows(), B.ncols());

    if constexpr (S == MatMulStrategy::Naive) {
        NaiveMatMulStrategy<TC>::multiply(A.getAccessor(), B.getAccessor(), result.getAccessor());
    }
    else if constexpr (S == MatMulStrategy::Shmem) {
        ShmemMatMulStrategy<TC>::multiply(A.getAccessor(), B.getAccessor(), result.getAccessor());
    }
    else if constexpr (S == MatMulStrategy::WMMA) {
        WMMAMatMulStrategy<TA, TB, TC>::multiply(A.getAccessor(), B.getAccessor(), result.getAccessor());
    }

    cuda_utils::checkCudaKernelErrors();

    return result;
}

