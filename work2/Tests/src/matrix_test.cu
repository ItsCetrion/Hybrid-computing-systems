#include <gtest/gtest.h>
#include <matrix.cuh>
#include <memory>

class MatrixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    matrix = std::make_unique<Matrix<float>>(rows, cols);
  }

  const std::size_t rows = 3;
  const std::size_t cols = 4;
  const std::size_t size = rows * cols;
  std::unique_ptr<Matrix<float>> matrix;
};

TEST_F(MatrixTest, ConstructorAndCols)
{
    EXPECT_EQ(matrix->ncols(), cols);
}

TEST_F(MatrixTest, ConstructorAndRows)
{
    EXPECT_EQ(matrix->nrows(), rows);
}

TEST_F(MatrixTest, ConstructorAndSize)
{
    EXPECT_EQ(matrix->size(), size);
}

TEST_F(MatrixTest, BlockAccess)
{
    auto block = matrix->getDeviceMemoryBlock();
    EXPECT_EQ(block.getSize(), size);

    const auto& const_matrix = matrix;
    auto const_block = const_matrix->getDeviceMemoryBlock();
    EXPECT_EQ(const_block.getSize(), size);
}

TEST_F(MatrixTest, AccessorAccess)
{
    auto accessor = matrix->getAccessor();
    EXPECT_EQ(accessor.size(), size);

    const auto& const_matrix = matrix;
    auto const_accessor = const_matrix->getAccessor();
    EXPECT_EQ(const_accessor.size(), size);
}

TEST_F(MatrixTest, AccessorDataConsistency)
{
    auto block = matrix->getDeviceMemoryBlock();
    auto accessor = matrix->getAccessor();
    EXPECT_EQ(block.getSize(), accessor.size());
}

template <typename T>
class MatrixTypedTest : public ::testing::Test {
    protected:
        const std::size_t rows = 3;
        const std::size_t cols = 3;
        const std::size_t size = rows * cols;
};

using TestTypes = ::testing::Types<float, int, double>;
TYPED_TEST_SUITE(MatrixTypedTest, TestTypes);

TYPED_TEST(MatrixTypedTest, DifferentTypes) {
  Matrix<TypeParam> matrix(this->rows, this->cols);
  EXPECT_EQ(matrix.size(), this->size);
}

TEST(MatrixTestInstance, InstanceIndependence) {
  const std::size_t rows1 = 3;
  const std::size_t rows2 = 4;
  const std::size_t cols1 = 3;
  const std::size_t cols2 = 4;

  Matrix<float> matrix1(rows1, cols1);
  Matrix<float> matrix2(rows2, cols2);

  EXPECT_EQ(matrix1.size(), rows1 * cols1);
  EXPECT_EQ(matrix2.size(), rows2 * cols2);
  EXPECT_NE(matrix1.size(), matrix2.size());
}

class MatrixSizeTest : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {};

TEST_P(MatrixSizeTest, SizeParameterized) {
  auto [rows, cols, size] = GetParam();
  Matrix<int> matrix(rows, cols);

  EXPECT_EQ(matrix.size(), size);
}

INSTANTIATE_TEST_SUITE_P(
    MatrixSizes, MatrixSizeTest, ::testing::Values(
         // Квадратные матрицы
        std::make_tuple(1, 1, 1),
        std::make_tuple(2, 2, 4),
        std::make_tuple(3, 3, 9),
        std::make_tuple(16, 16, 256),
        // Прямоугольные матрицы
        std::make_tuple(2, 3, 6),
        std::make_tuple(5, 3, 15),
        // Граничные случаи
        std::make_tuple(127, 128, 16256),
        std::make_tuple(128, 128, 16384)
    ));
