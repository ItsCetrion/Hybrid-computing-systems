#ifndef KERNEL_MATRIX_MULTIPLY_HPP
#define KERNEL_MATRIX_MULTIPLY_HPP

#include <cuda_runtime.h>
#include <matrix.cuh>


template<typename T, bool transposeA, bool transposeB>
__global__ void kernelMatrixMultiply(const T *A, const T *B, const T *bias, T *C, std::size_t rowsA, std::size_t colsA, std::size_t colsB) {

    std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < rowsA && j < colsB) {
        T sum = 0;
        for (std::size_t k = 0; k < colsA; ++k) {
            T a = transposeA ? A[k * rowsA + i] : A[i * colsA + k];
            T b = transposeB ? B[j * colsA + k] : B[k * colsB + j];
            sum += a * b;
        }
        C[i * colsB + j] = sum + bias[i];
    }

}


#endif // KERNEL_MATRIX_MULTIPLY_HPP

