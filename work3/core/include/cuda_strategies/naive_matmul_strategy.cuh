#pragma once

#include "../utils/cuda_utils/cuda_utils.cuh"
#include "../kernels/kernel_matrix_multiply.cuh"


template <class T>
class NaiveMatMulStrategy {
    public:
        void static multiply(const MatrixAccessor<T>& A, const MatrixAccessor<T>& B, MatrixAccessor<T>& C) {

            constexpr std::size_t blockSize = 16;
            auto [blocks, threads] = cuda_utils::calcGridSize(blockSize, C.nrows(), C.ncols());
            
            kernel_matmul_naive<T><<<blocks, threads>>>(A, B, C);

        }
};

