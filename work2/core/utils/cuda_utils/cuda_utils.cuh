#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <utility>
#include <cuda_runtime.h>


namespace cuda_utils {

    std::pair<dim3, dim3> calcGridSize(std::size_t blockSize, std::size_t workSizeRows, std::size_t workSizeCols) {
        dim3 blocks((workSizeCols + blockSize - 1) / blockSize,
                    (workSizeRows + blockSize - 1) / blockSize);
        dim3 threads(blockSize, blockSize);

        return {blocks, threads};
    };

}

#endif // CUDA_UTILS_HPP