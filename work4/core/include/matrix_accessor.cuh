#pragma once

#include <cuda_runtime.h>


template <typename T>
class MatrixAccessor {
    private:
        T *data;
        std::size_t numRows;
        std::size_t numCols;
        std::size_t stride_;

    public:
        __host__ __device__ MatrixAccessor(T *dataPtr, std::size_t nrows, std::size_t ncols, std::size_t stride)
            : data(dataPtr),
              numRows(nrows),
              numCols(ncols),
              stride_(stride) {}

        __host__ __device__ std::size_t nrows() const {
            return this->numRows;
        }

        __host__ __device__ std::size_t ncols() const {
            return this->numCols;
        }

        __host__ __device__ std::size_t size() const {
            return this->numRows * this->numCols;
        }

        __host__ __device__ std::size_t stride() const {
            return this->stride_;
        }

        __device__ T& operator()(std::size_t i, std::size_t j) {
            return this->data[i * this->stride_ + j];
        }

        __device__ const T& operator()(std::size_t i, std::size_t j) const {
            return this->data[i * this->stride_ + j];
        }

        __device__ T& operator[](std::size_t n) {
            return this->data[n];
        }

        __device__ const T& operator[](std::size_t n) const {
            return this->data[n];
        }

};
