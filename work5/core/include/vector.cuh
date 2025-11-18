#pragma once

#include <memory>

#include "device_memory_block.cuh"
#include "vector_accessor.cuh"
#include "../utils/cuda_utils/cuda_utils.cuh"


template <class T>
class Vector {
    private:
        std::shared_ptr<DeviceMemoryBlock<T>> deviceMemoryBlock_;
        VectorAccessor<T> accessor_;

    public:
        Vector(std::size_t size)
            : deviceMemoryBlock_(std::make_shared<DeviceMemoryBlock<T>>(size)),
            accessor_(deviceMemoryBlock_->data(), size) {}

        std::size_t size() const {
            return accessor_.size();
        }

        DeviceMemoryBlock<T>& deviceMemoryBlock() {
            return *deviceMemoryBlock_;
        }

        const DeviceMemoryBlock<T>& deviceMemoryBlock() const {
            return *deviceMemoryBlock_;
        }

        VectorAccessor<T>& accessor() {
            return accessor_;
        }

        const VectorAccessor<T>& accessor() const {
            return accessor_;
        }
};
