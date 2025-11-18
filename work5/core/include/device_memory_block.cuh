#pragma once

#include <cuda_runtime.h>

#include <string>
#include <iostream>


template <typename T>
class DeviceMemoryBlock {
    private:
        T *data_;
        std::size_t size_;

        inline void checkCudaError(const cudaError_t &error) const {
            if (error != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
            }
        }

        inline void checkCudaErrorNoExcept(const cudaError_t &error) const {
            if (error != cudaSuccess) {
                std::cerr << std::string("CUDA error: ") + cudaGetErrorString(error) << std::endl;
            }
        }

    public:
        DeviceMemoryBlock(std::size_t size)
            : size_(size),
            data_(nullptr) {
            if (size <= 0) {
                throw std::invalid_argument("The size of DeviceMemoryBlock must be a non-negative integer");
            }
            checkCudaError(cudaMalloc(&data_, size_ * sizeof(T)));
        }

        DeviceMemoryBlock(const DeviceMemoryBlock &other)
            : size_(other.size_),
            data_(nullptr) {
            checkCudaError(cudaMalloc(&data_, size_ * sizeof(T)));
            checkCudaError(cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        DeviceMemoryBlock(DeviceMemoryBlock &&other) noexcept
            : data_(other.data_),
            size_(other.size_) {
            other.data_ = nullptr;
            other.size_ = 0;
        }

        DeviceMemoryBlock& operator=(const DeviceMemoryBlock &other) {
            if (this != &other) {
                if (data_) {
                    checkCudaError(cudaFree(data_));
                }
                size_ = other.size_;
                checkCudaError(cudaMalloc(&data_, size_ * sizeof(T)));
                checkCudaError(cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
            }
            return *this;
        }

        DeviceMemoryBlock& operator=(DeviceMemoryBlock&& other) noexcept {
            if (this != &other) {
                if (data_) {
                    checkCudaErrorNoExcept(cudaFree(data_));
                }
                size_ = other.size_;
                data_ = other.data_;
                other.data_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        T* data() {
            return data_;
        }

        const T* data() const {
            return data_;
        }

        std::size_t size() const {
            return size_;
        }

        void copyToHost(T *hostPtr) const {
            checkCudaError(cudaMemcpy(hostPtr, data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }

        void copyFromHost(const T *hostPtr) {
            checkCudaError(cudaMemcpy(data_, hostPtr, size_ * sizeof(T), cudaMemcpyHostToDevice));
        }

        ~DeviceMemoryBlock() {
            if (data_) {
                checkCudaErrorNoExcept(cudaFree(data_));
            }
        }
};
