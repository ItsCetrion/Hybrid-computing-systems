#pragma once

#include <memory>

#include "device_memory_block.cuh"
#include "matrix_accessor.cuh"
#include "./kernels/kernel_matrix_multiply.cuh"
#include "../utils/cuda_utils/cuda_utils.cuh"
#include "./cuda_strategies/matmul_strategy_interface.cuh"


template <typename T>
class Matrix {
    private:
        std::shared_ptr<DeviceMemoryBlock<T>> deviceMemoryBlock;
        MatrixAccessor<T> accessor;
        std::unique_ptr<IMatMulStrategy<T>> mulStrategy;

        static auto createDeviceMemoryBlock(std::size_t nrows, std::size_t ncols) {
            if (nrows <= 0 || ncols <= 0) {
                throw std::invalid_argument("The dimensions of the matrix must be a non-negative integer");
            }
            return std::make_shared<DeviceMemoryBlock<T>>(nrows * ncols);
        }

    public:
        Matrix(std::size_t nrows, std::size_t ncols)
            : deviceMemoryBlock(this->createDeviceMemoryBlock(nrows, ncols)), 
              accessor(this->deviceMemoryBlock->getData(), nrows, ncols, ncols) {}

        std::size_t nrows() const {
            return this->accessor.nrows();
        }

        std::size_t ncols() const {
            return this->accessor.ncols();
        }

        std::size_t size() const {
            return this->accessor.size();
        }

        DeviceMemoryBlock<T> &getDeviceMemoryBlock() {
            return *this->deviceMemoryBlock;
        }

        const DeviceMemoryBlock<T> &getDeviceMemoryBlock() const {
            return *this->deviceMemoryBlock;
        }

        MatrixAccessor<T>& getAccessor() {
            return this->accessor;
        }

        const MatrixAccessor<T>& getAccessor() const {
            return this->accessor;
        }

        Matrix<T> multiply(const Matrix<T> &other, const IMatMulStrategy<T>& mulStrategy) const {
            if (this->ncols() != other.nrows()) {
                throw std::runtime_error("Matrices are not compatible for multiplication");
            }

            Matrix<T> result(this->nrows(), other.ncols());

            mulStrategy.multiply(this->accessor, other.accessor, result.accessor);

            cuda_utils::checkCudaKernelErrors();
            return result;

        }

};

