#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <matrix.cuh>
#include "matrix_operations.cuh"
#include "cuda_strategies/wmma_matmul_strategy.cuh"

#define EIGEN_NO_CUDA


class WMMAMatrixMulTest : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t, float>> {
    protected:
    bool matmul_test_impl(std::size_t rows_a, std::size_t cols_a, std::size_t cols_b, float tol) {
        using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        RowMatrixXf a_target = RowMatrixXf::Random(rows_a, cols_a);
        RowMatrixXf b_target = RowMatrixXf::Random(cols_a, cols_b);
        
        RowMatrixXf c_target = a_target * b_target;
        
        Matrix<float> a(rows_a, cols_a);
        a.getDeviceMemoryBlock().copyFromHost(a_target.data());
        
        Matrix<float> b(cols_a, cols_b);
        b.getDeviceMemoryBlock().copyFromHost(b_target.data());
        
        Matrix<float> result = matrix_ops::multiply<float, WMMAMatMulStrategy>(a, b);

        cudaDeviceSynchronize(); 
        
        if (result.nrows() != rows_a || result.ncols() != cols_b) return false;
        
        RowMatrixXf c_from_device = RowMatrixXf::Zero(rows_a, cols_b);
        result.getDeviceMemoryBlock().copyToHost(c_from_device.data());
        
        return c_target.isApprox(c_from_device, tol);
    }
};

TEST_P(WMMAMatrixMulTest, matmul_test) {
  auto [rows_a, cols_a, cols_b, tol] = GetParam();
  EXPECT_TRUE(matmul_test_impl(rows_a, cols_a, cols_b, tol));
}

INSTANTIATE_TEST_SUITE_P(
    WMMAMatrixMulTestSuite,
    WMMAMatrixMulTest,
    ::testing::Values(
        // Квадратные матрицы
        std::make_tuple(16, 16, 16, 1e-5),
        std::make_tuple(32, 32, 32, 1e-5),
        std::make_tuple(48, 48, 48, 1e-5),
        std::make_tuple(64, 64, 64, 1e-5),
        // Прямоугольные матрицы
        std::make_tuple(32, 48, 64, 1e-5),
        std::make_tuple(80, 48, 112, 1e-5),
        // Граничные случаи
        std::make_tuple(128, 144, 160, 1e-5),
        std::make_tuple(128, 128, 128, 1e-5),
        std::make_tuple(512, 512, 512, 1e-5)
    )
);