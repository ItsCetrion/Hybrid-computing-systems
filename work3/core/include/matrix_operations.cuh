#pragma once

#include <cuda_runtime.h>

#include "./kernels/kernel_matrix_multiply.cuh"
#include "matrix_accessor.cuh"
#include "matrix.cuh"


namespace matrix_ops {

    template <class T, template<class> class MatMulStrategy>
    Matrix<T> multiply(const Matrix<T>& A, const Matrix<T>& B) {
        if (A.ncols() != B.nrows()) {
            throw std::runtime_error("Matrices are not compatible for multiplication");
        }
        Matrix<T> result(A.nrows(), B.ncols());

        MatMulStrategy<T>::multiply(A.getAccessor(), B.getAccessor(), result.getAccessor());

        cuda_utils::checkCudaKernelErrors();
        return result;
    }

}
