#ifndef _CUDAGH_HPP_
#define _CUDAGH_HPP_

#include <cstdint>
#include <cuda_runtime.h>

namespace cudagh {

inline std::pair<dim3, dim3> cover(std::size_t blockSize, std::size_t workSizeRows, std::size_t workSizeCols) {
        dim3 blocks((workSizeCols + blockSize - 1) / blockSize,
                    (workSizeRows + blockSize - 1) / blockSize);
        dim3 threads(blockSize, blockSize);

        return {blocks, threads};
    };


}  // namespace cudagh

#endif  // _CUDAGH_HPP_