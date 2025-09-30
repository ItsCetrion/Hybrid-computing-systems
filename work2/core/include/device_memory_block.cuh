#ifndef DEVICE_ARRAY_HPP
#define DEVICE_ARRAY_HPP

#include <stdlib.h>
#include <cuda_runtime.h>


template<typename T>
class DeviceMemoryBlock {
    private:
        T *data;
        std::size_t size;

    public:
        DeviceMemoryBlock(std::size_t sizeVal) : size(sizeVal), data(nullptr) {
            cudaMalloc(&this->data, this->size * sizeof(T));
        }

        DeviceMemoryBlock(DeviceMemoryBlock&& other) noexcept
            : data(other.data), size(other.size) {
            other.data = nullptr;
            other.size = 0;
        }

        T* getData() const {
            return this->data;
        }

        std::size_t getSize() const {
            return this->size;
        }

        void copyToHost(T *hostPtr) const {
            cudaMemcpy(hostPtr, this->data, this->size * sizeof(T), cudaMemcpyDeviceToHost);
        }

        void copyFromHost(const T *hostPtr) {
            cudaMemcpy(this->data, hostPtr, this->size * sizeof(T), cudaMemcpyHostToDevice);
        }

        ~DeviceMemoryBlock() {
            if (this->data) {
                cudaFree(this->data);
            }
        }

};

#endif // DEVICE_ARRAY_HPP

