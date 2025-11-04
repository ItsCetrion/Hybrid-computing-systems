#pragma once

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#include "matrix_accessor.cuh"


template <class TA, class TB, class TC>
__global__ void kernel_matmul_wmma(const MatrixAccessor<TA> A, const MatrixAccessor<TB> B, MatrixAccessor<TC> C) {

    using namespace nvcuda;

    constexpr size_t wmmaM = 16;
    constexpr size_t wmmaN = 16;
    constexpr size_t wmmaK = 16;
    constexpr size_t warpSize = 32;

    size_t warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    size_t warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    size_t aRow = warpM * wmmaM;
    size_t bCol = warpN * wmmaN;

    if (aRow >= C.nrows() || bCol >= C.ncols()) return;

    wmma::fragment<wmma::matrix_a, wmmaM, wmmaN, wmmaK, TA, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, wmmaM, wmmaN, wmmaK, TB, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, wmmaM, wmmaN, wmmaK, TC> cFrag;

    wmma::fill_fragment(cFrag, 0.0f);

    for (size_t k = 0; k < A.ncols(); k += 16) {
        size_t aCol = k;
        size_t bRow = k;
        
        wmma::load_matrix_sync(aFrag, &A(aRow, aCol), A.stride());
        wmma::load_matrix_sync(bFrag, &B(bRow, bCol), B.stride());

        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);

    }

    wmma::store_matrix_sync(&C(aRow, bCol), cFrag, C.stride(), wmma::mem_row_major);
    
}

