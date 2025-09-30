#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include <matrix.cuh>
#include <kernel_matrix_multiply.cuh>
#include <stdexcept>


class MatrixOperations {
    public:
        template<typename T, bool transposeA, bool transposeB>
        static Matrix<T> matrixMultiplyAddBias(const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> &b) {

            std::size_t rowsA = transposeA ? A.cols() : A.rows();
            std::size_t colsA = transposeA ? A.rows() : A.cols();
            std::size_t rowsB = transposeB ? B.cols() : B.rows();
            std::size_t colsB = transposeB ? B.rows() : B.cols();

            if (colsA != rowsB) {
                throw std::runtime_error("Matrices are not compatible for multiplication");
            }

            if (b.rows() != rowsA || b.cols() > 1) {
                throw std::runtime_error("The column vector does not match in dimension");
            }

            Matrix<T> result(rowsA, colsB);

            constexpr std::size_t BLOCK_SIZE = 16;

            dim3 blocks((colsB + BLOCK_SIZE - 1) / BLOCK_SIZE, (rowsA + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

            kernelMatrixMultiply<T, transposeA, transposeB><<<blocks, threads>>>(A.data(), B.data(), b.data(), result.data(), rowsA, colsA, colsB);

            return result;

        }

};


#endif // MATRIX_OPERATIONS

