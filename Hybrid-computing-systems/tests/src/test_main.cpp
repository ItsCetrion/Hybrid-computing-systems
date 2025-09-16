#include <gtest/gtest.h>
#include "CPU_operations.h"
#include "cuda_operations.h"
#include <cuda_runtime.h>
#include <cstdlib>

void fill_a_b(float *host_a, float *host_b, const int vec_size);

TEST(CoreTest_GPU, AddVec){
  const int N = 256;
  size_t size = N * sizeof(float);
  float* host_a = (float*) malloc(size);
  float* host_b = (float*) malloc(size);
  float* host_res_CPU = (float*) malloc(size);
  float* host_res_GPU = (float*) malloc(size);

  fill_a_b(host_a, host_b, N);
  addVectorsCPU(host_a, host_b, host_res_CPU, N);
  addVectorsOptimal(host_a, host_b, host_res_GPU, N);

  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(host_res_CPU[i], host_res_GPU[i]);
  }

  free(host_a);
  free(host_b);
  free(host_res_CPU);
  free(host_res_GPU);
}

void fill_a_b(float *host_a, float *host_b, const int vec_size){
  for (int i = 0; i < vec_size; i++) {
    host_a[i] = (float)i;
    host_b[i] = (float)i;
  }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}