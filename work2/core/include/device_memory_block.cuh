#ifndef DEVICE_MEMORY_BLOCK_HPP
#define DEVICE_MEMORY_BLOCK_HPP

#include <cuda_runtime.h>

#include <string>


template <typename T>
class DeviceMemoryBlock {
    private:
        T *data;
        std::size_t size;

        inline void checkCudaError(const cudaError_t &error) const {
            if (error != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
            }
        }

    public:
        DeviceMemoryBlock(std::size_t sizeVal)
            : size(sizeVal),
            data(nullptr) {
            checkCudaError(cudaMalloc(&this->data, this->size * sizeof(T)));
        }

        DeviceMemoryBlock(const DeviceMemoryBlock &other)
            : size(other.size),
            data(nullptr) {
            checkCudaError(cudaMalloc(&this->data, this->size * sizeof(T)));
            checkCudaError(cudaMemcpy(this->data, other.data, this->size * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        DeviceMemoryBlock(DeviceMemoryBlock &&other) noexcept
            : data(other.data),
            size(other.size) {
            other.data = nullptr;
            other.size = 0;
        }

        DeviceMemoryBlock& operator=(const DeviceMemoryBlock &other) {
            if (this != other) {
                if (this->data) {
                    checkCudaError(cudaFree(this->data));
                }
                this->size = other.size;
                checkCudaError(cudaMalloc(&this->data, this->size * sizeof(T)));
                checkCudaError(cudaMemcpy(this->data, other.data, this->size * sizeof(T), cudaMemcpyDeviceToDevice));
            }
            return *this;
        }

        DeviceMemoryBlock& operator=(DeviceMemoryBlock&& other) noexcept {
            if (this != &other) {
                if (this->data) {
                    checkCudaError(cudaFree(this->data));
                }
                this->size = other.size;
                this->data = other.data;
                other->data = nullptr;
                other.size = 0;
            }
            return *this;
        }

        T* getData() const {
            return this->data;
        }

        std::size_t getSize() const {
            return this->size;
        }

        void copyToHost(T *hostPtr) const {
            checkCudaError(cudaMemcpy(hostPtr, this->data, this->size * sizeof(T), cudaMemcpyDeviceToHost));
        }

        void copyFromHost(const T *hostPtr) {
            checkCudaError(cudaMemcpy(this->data, hostPtr, this->size * sizeof(T), cudaMemcpyHostToDevice));
        }

        ~DeviceMemoryBlock() {
            if (this->data) {
                checkCudaError(cudaFree(this->data));
            }
        }
};

#endif // DEVICE_MEMORY_BLOCK_HPP

