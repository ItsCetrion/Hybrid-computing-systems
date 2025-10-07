#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <memory>

#include <device_memory_block.cuh>
#include <matrix_accessor.cuh>
#include <kernel_matrix_multiply.cuh>
#include <../utils/cuda_utils/cuda_utils.cuh>


template <typename T>
class Matrix {
    private:
        std::shared_ptr<DeviceMemoryBlock<T>> deviceMemoryBlock;
        MatrixAccessor<T> accessor;

        static auto createDeviceMemoryBlock(std::size_t nrows, std::size_t ncols) {
            if (nrows <= 0 || ncols <= 0) {
                throw std::invalid_argument("The dimensions of the matrix must be a non-negative integer");
            }
            return std::make_shared<DeviceMemoryBlock<T>>(nrows * ncols);
        }

    public:
        Matrix(std::size_t nrows, std::size_t ncols)
            : deviceMemoryBlock(this->createDeviceMemoryBlock(nrows, ncols)), 
              accessor(this->deviceMemoryBlock->getData(), nrows, ncols) {}

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

        Matrix<T> operator*(const Matrix<T> &rhs) const {

            if (this->ncols() != rhs.nrows()) {
                throw std::runtime_error("Matrices are not compatible for multiplication");
            }

            Matrix<T> result(this->nrows(), rhs.ncols());

            constexpr std::size_t BLOCK_SIZE = 16;
            auto [blocks, threads] = cuda_utils::calcGridSize(BLOCK_SIZE, result.nrows(), result.ncols());

            kernel_matmul_naive<T><<<blocks, threads>>>(this->accessor, rhs.accessor, result.accessor);

            cuda_utils::checkCudaKernelErrors();

            return result;
        }

};

#endif // MATRIX_HPP

