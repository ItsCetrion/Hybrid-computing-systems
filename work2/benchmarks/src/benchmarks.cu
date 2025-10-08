#include <cudagh.hpp>
#include <kernel_matrix_multiply.cuh>
#include <matrix.cuh>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_timer.hpp> 