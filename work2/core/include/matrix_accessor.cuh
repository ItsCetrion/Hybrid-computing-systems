#ifndef MATRIX_ACCESSOR_HPP
#define MATRIX_ACCESSOR_HPP

#include <cuda_runtime.h>


template <typename T>
class MatrixAccessor {
    private:
        T *data;
        std::size_t numRows;
        std::size_t numCols;

    public:
        __host__ __device__ MatrixAccessor(T *dataPtr, std::size_t rowsVal, std::size_t colsVal)
            : data(dataPtr),
              numRows(rowsVal),
              numCols(colsVal) {}

        __host__ __device__ std::size_t rows() const {
            return this->numRows;
        }

        __host__ __device__ std::size_t cols() const {
            return this->numCols;
        }

        __host__ __device__ std::size_t numElements() const {
            return this->numRows * this->numCols;
        }

        __host__ __device__ T& operator()(std::size_t i, std::size_t j) {
            return this->data[i * this->numCols + j];
        }

        __host__ __device__ const T& operator()(std::size_t i, std::size_t j) const {
            return this->data[i * this->numCols + j];
        }
};

#endif // MATRIX_ACCESSOR_HPP

