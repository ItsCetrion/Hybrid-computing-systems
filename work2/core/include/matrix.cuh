#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <device_memory_block.cuh>
#include <matrix_accessor.cuh>
#include <kernel_matrix_multiply.cuh>
#include <../utils/cuda_utils/cuda_utils.cuh>

#include <memory>


template <typename T>
class Matrix {
    private:
        DeviceMemoryBlock<T> deviceMemoryBlock;
        std::size_t numRows;
        std::size_t numCols;

    public:
        Matrix(std::size_t rowsVal, std::size_t colsVal)
            : deviceMemoryBlock(rowsVal * colsVal), numRows(rowsVal), numCols(colsVal) {}

        MatrixAccessor<T> createAccessor() {
            return MatrixAccessor<T>(this->deviceMemoryBlock.getData(), this->numRows, this->numCols);
        }

        const MatrixAccessor<T> createAccessor() const {
            return MatrixAccessor<T>(this->deviceMemoryBlock.getData(), this->numRows, this->numCols);
        }

        DeviceMemoryBlock<T> &getDeviceMemoryBlock() {
            return this->deviceMemoryBlock;
        }

        const DeviceMemoryBlock<T> &getDeviceMemoryBlock() const {
            return this->deviceMemoryBlock;
        }

        Matrix<T> operator*(const Matrix<T> &rhs) const {
            MatrixAccessor<T> leftMatrix = this->createAccessor();
            MatrixAccessor<T> rightMatrix = rhs.createAccessor();

            if (leftMatrix.cols() != rightMatrix.rows()) {
                throw std::runtime_error("Matrices are not compatible for multiplication");
            }

            Matrix<T> result(leftMatrix.rows(), rightMatrix.cols());
            MatrixAccessor<T> resultAccessor = result.createAccessor();

            constexpr std::size_t BLOCK_SIZE = 16;
            auto [blocks, threads] = cuda_utils::calcGridSize(BLOCK_SIZE, resultAccessor.rows(), resultAccessor.cols());

            kernel_matmul_naive<T><<<blocks, threads>>>(leftMatrix, rightMatrix, resultAccessor);

            return result;
        }

};

#endif // MATRIX_HPP

