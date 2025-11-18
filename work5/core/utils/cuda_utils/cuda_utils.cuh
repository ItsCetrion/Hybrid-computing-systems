#pragma once

#include <utility>
#include <cuda_runtime.h>


namespace cuda_utils {

    inline std::size_t calcGridSize(std::size_t vectorSize, std::size_t blockSize, std::size_t unrollFactor) {
        std::size_t elementsPerBlock = blockSize * unrollFactor;
        return (vectorSize + elementsPerBlock - 1) / elementsPerBlock;
    }

    inline void checkCudaKernelErrors() {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA kernel execution failed: ") + cudaGetErrorString(err));
        }
    }

}
