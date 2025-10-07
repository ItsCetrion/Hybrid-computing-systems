#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <../core/include/matrix.cuh>

#define EIGEN_NO_CUDA

class MatrixMulTest : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t, float>> {
    protected:
    bool matmul_test_impl(std::size_t rows_a, std::size_t cols_a, std::size_t cols_b, float tol) {
        Eigen::MatrixXf a_target = Eigen::MatrixXf::Random(rows_a, cols_a);
        Eigen::MatrixXf b_target = Eigen::MatrixXf::Random(cols_a, cols_b);
        
        Eigen::MatrixXf c_target = a_target * b_target;
        
        Matrix<float> a(rows_a, cols_a);
        a.getDeviceMemoryBlock().copyFromHost(a_target.data());
        
        Matrix<float> b(cols_a, cols_b);
        b.getDeviceMemoryBlock().copyFromHost(b_target.data());
        
        Matrix<float> c = a * b;
        
        if (c.rows() != rows_a || c.cols() != cols_b) return false;
        
        Eigen::MatrixXf c_from_device = Eigen::MatrixXf::Zero(rows_a, cols_b);
        c.getDeviceMemoryBlock().copyToHost(c_from_device.data());
        
        return c_target.isApprox(c_from_device, tol);
    }
};

TEST_P(MatrixMulTest, matmul_test) {
  auto [rows_a, cols_a, cols_b, tol] = GetParam();
  EXPECT_TRUE(matmul_test_impl(rows_a, cols_a, cols_b, tol));
}

INSTANTIATE_TEST_SUITE_P(
    MatrixMulTestSuite,
    MatrixMulTest,
    ::testing::Values(
        // Квадратные матрицы
        std::make_tuple(1, 1, 1, 1e-6),
        std::make_tuple(2, 2, 2, 1e-6),
        std::make_tuple(3, 3, 3, 1e-6),
        std::make_tuple(16, 16, 16, 1e-6),
        // Прямоугольные матрицы
        std::make_tuple(2, 3, 4, 1e-6),
        std::make_tuple(5, 3, 7, 1e-6),
        // Граничные случаи
        std::make_tuple(127, 128, 129, 1e-6),
        std::make_tuple(128, 128, 128, 1e-6),
        std::make_tuple(256, 256, 256, 1e-6)
    )
);
