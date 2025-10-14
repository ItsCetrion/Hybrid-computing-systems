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

template <typename T, std::size_t blockSize>
__global__ void kernel_matmul_shmem(const MatrixAccessor<T> A, const MatrixAccessor<T> B, MatrixAccessor<T> C) {

    std::size_t blockRow = blockIdx.y;
    std::size_t blockCol = blockIdx.x;

    std::size_t localThreadRow = threadIdx.y;
    std::size_t localThreadCol = threadIdx.x;

    std::size_t globalThreadRow = blockRow * blockSize + localThreadRow;
    std::size_t globalThreadCol = blockCol * blockSize + localThreadCol;

    std::size_t numTiles = (A.ncols() + blockSize - 1) / blockSize;

    T sum = 0;

    for (std::size_t m = 0; m < numTiles; ++m) {

        __shared__ T As[blockSize][blockSize];
        __shared__ T Bs[blockSize][blockSize];
        
        std::size_t Acol = m * blockSize + localThreadCol;
        std::size_t Brow = m * blockSize + localThreadRow;

        if (globalThreadRow < A.nrows() && Acol < A.ncols()) {
            As[localThreadRow][localThreadCol] = A(globalThreadRow, Acol);
        }
        else {
            As[localThreadRow][localThreadCol] = 0;
        }
        
        if (Brow < B.nrows() && globalThreadCol < B.ncols()) {
            Bs[localThreadRow][localThreadCol] = B(Brow, globalThreadCol);
        }
        else {
            Bs[localThreadRow][localThreadCol] = 0;
        }

        __syncthreads();
        
        for (std::size_t k = 0; k < blockSize; ++k) {
            sum += As[localThreadRow][k] * Bs[k][localThreadCol];
        }
        __syncthreads();
    }

    if (globalThreadRow < C.nrows() && globalThreadCol < C.ncols()) {
        C(globalThreadRow, globalThreadCol) = sum;
    }

}

