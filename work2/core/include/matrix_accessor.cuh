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
        __host__ __device__ MatrixAccessor(T *dataPtr, std::size_t nrows, std::size_t ncols)
            : data(dataPtr),
              numRows(nrows),
              numCols(ncols) {}

        __host__ __device__ std::size_t nrows() const {
            return this->numRows;
        }

        __host__ __device__ std::size_t ncols() const {
            return this->numCols;
        }

        __host__ __device__ std::size_t size() const {
            return this->numRows * this->numCols;
        }

        __host__ __device__ T& operator()(std::size_t i, std::size_t j) {
            return this->data[i * this->numCols + j];
        }

        __host__ __device__ const T& operator()(std::size_t i, std::size_t j) const {
            return this->data[i * this->numCols + j];
        }

        __host__ __device__ T& operator[](std::size_t n) {
            return this->data[n];
        }

        __host__ __device__ const T& operator[](std::size_t n) const {
            return this->data[n];
        }

};

#endif // MATRIX_ACCESSOR_HPP

