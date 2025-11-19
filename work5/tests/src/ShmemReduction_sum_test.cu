#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "vector.cuh"
#include "vector_operations.cuh"

#define EIGEN_NO_CUDA


class ShmemReductionSumTest : public ::testing::TestWithParam<std::tuple<std::size_t, float>> {
    protected:
    bool matsum_test_impl(std::size_t size, float tol) {
        
        Eigen::VectorXf en_vector = Eigen::VectorXf::Random(size);
        float en_vector_result = en_vector.sum();

        Vector<float> my_vector(size);
        my_vector.deviceMemoryBlock().copyFromHost(en_vector.data());
        float my_vector_result = sum(my_vector, ShmemReductionTag{});
        EXPECT_NEAR(en_vector_result, my_vector_result, tol);
    }
};

TEST_P(ShmemReductionSumTest, matsum_test) {
  auto [size, tol] = GetParam();
  matsum_test_impl(size, tol);
}

INSTANTIATE_TEST_SUITE_P(
    ShmemReductionSumTestSuite,
    ShmemReductionSumTest,
    ::testing::Values(
        std::make_tuple(1, 1e-4),
        std::make_tuple(2, 1e-4),
        std::make_tuple(3, 1e-4),
        std::make_tuple(127, 1e-4),
        std::make_tuple(129, 1e-4),
        std::make_tuple(512, 1e-4),
        std::make_tuple(541, 1e-4),
        std::make_tuple(1037, 1e-4)
    )
);