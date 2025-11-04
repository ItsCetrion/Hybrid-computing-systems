#pragma once

#include <utility>
#include <cuda_runtime.h>


namespace cuda_utils {

    inline std::pair<dim3, dim3> calcGridSize(std::size_t blockSize, std::size_t workSizeRows, std::size_t workSizeCols) {
        dim3 blocks((workSizeCols + blockSize - 1) / blockSize,
                    (workSizeRows + blockSize - 1) / blockSize);
        dim3 threads(blockSize, blockSize);

        return {blocks, threads};
    };

    inline std::pair<dim3, dim3> calcGridSizeWMMA(size_t matrix_rows, size_t matrix_cols) {
        constexpr size_t warp_size = 32;
        constexpr size_t wmma_tile_size = 16;
        
        size_t warps_in_m = (matrix_rows + wmma_tile_size - 1) / wmma_tile_size;
        size_t warps_in_n = (matrix_cols + wmma_tile_size - 1) / wmma_tile_size;
        
        dim3 block_dim(128, 4);
        
        size_t warps_per_block_x = block_dim.x / warp_size;
        size_t warps_per_block_y = block_dim.y;
        
        dim3 grid_dim((warps_in_m + warps_per_block_x - 1) / warps_per_block_x,
                    (warps_in_n + warps_per_block_y - 1) / warps_per_block_y);
    
        return {grid_dim, block_dim};
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
