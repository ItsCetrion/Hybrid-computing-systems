#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <device_memory_block.cuh>


template <typename T>
class Matrix {
    private:
        DeviceMemoryBlock<T> deviceMemoryBlock;
        std::size_t numRows;
        std::size_t numCols;

    public:
        Matrix(std::size_t rowsVal, std::size_t colsVal)
            : deviceMemoryBlock(rowsVal * colsVal), numRows(rowsVal), numCols(colsVal) {}

        Matrix(Matrix&& other) noexcept
            : deviceMemoryBlock(std::move(other.deviceMemoryBlock)),
              numRows(other.numRows),
              numCols(other.numCols) {}

        T* data() const {
            return this->deviceMemoryBlock.getData();
        }

        DeviceMemoryBlock<T>& getDeviceMemoryBlock() {
            return this->deviceMemoryBlock;
        }

        const DeviceMemoryBlock<T>& getDeviceMemoryBlock() const {
            return this->deviceMemoryBlock;
        }

        std::size_t rows() {
            return this->numRows;
        }

        std::size_t cols() {
            return this->numCols;
        }

        std::size_t rows() const {
            return this->numRows;
        }

        std::size_t cols() const {
            return this->numCols;
        }

        __host__ __device__ std::size_t numElements() {
            return this->numRows * this->numCols;
        }

};


#endif // MATRIX_HPP