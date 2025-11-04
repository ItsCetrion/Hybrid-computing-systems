#pragma once

#include "../kernels/kernel_matmul_wmma.cuh"
#include "../utils/cuda_utils/cuda_utils.cuh"


template <class TA, class TB, class TC>
class WMMAMatMulStrategy {
    public:
        void static multiply(const MatrixAccessor<TA>& A, const MatrixAccessor<TB>& B, MatrixAccessor<TC>& C) {

            auto [gridDim, blockDim] = cuda_utils::calcGridSizeWMMA(C.nrows(), C.ncols());
            kernel_matmul_wmma<<<gridDim, blockDim>>>(A, B, C);

        }
};