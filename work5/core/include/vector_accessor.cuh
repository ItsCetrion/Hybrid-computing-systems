#pragma once

#include <cuda_runtime.h>


template <class T>
class VectorAccessor {
    private:
        T *data_;
        std::size_t size_;

    public:
        __host__ __device__ VectorAccessor(T *data, std::size_t size)
            : data_(data),
              size_(size) {}

        __host__ __device__ std::size_t size() const {
            return size_;
        }

        __host__ __device__ T& operator[](std::size_t n) {
            return this->data_[n];
        }

        __host__ __device__ const T& operator[](std::size_t n) const {
            return this->data_[n];
        }

};
