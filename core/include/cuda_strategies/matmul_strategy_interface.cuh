#pragma once

#include "../matrix_accessor.cuh"


template <class T>
class IMatMulStrategy {
    public:
        virtual void multiply(const MatrixAccessor<T> A, const MatrixAccessor<T> B, MatrixAccessor<T> C) const = 0;
        virtual ~IMatMulStrategy() = default;
};

