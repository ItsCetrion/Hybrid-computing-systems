#include <gtest/gtest.h>
#include "Core.cuh"
#include <cuda_runtime.h>

 float* gpu();

TEST(CoreTest_CPU, SumArray) {
    int data[] = { 1, 2, 3, 4, 5 }; 
    EXPECT_EQ(sumArray(data, 5), 15);
}

TEST(CoreTest_GPU, addVec){
  const int N = 256;
  size_t size = N * sizeof(float);
  float* host_c = (float*) malloc(size);
  float* host_c_res = (float*) malloc(size);

  for (int i = 0; i < N; i++) {
    *(host_c + i) = (float)i*2;
  }

  host_c_res = gpu();
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(host_c_res[i], host_c[i]);
  }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

 float* gpu(){
  const int N = 256;
  size_t size = N * sizeof(float);
  float* host_a = (float*) malloc(size);
  float* host_b = (float*) malloc(size);
  float* host_c = (float*) malloc(size);
  // float* host_c_cpu = (float*) malloc(size);

  for (int i = 0; i < N; i++) {
    host_a[i] = i;
    host_b[i] = i;
  }

  float* device_a;
  float* device_b;
  float* device_c;

  cudaMalloc(&device_a, size);
  cudaMalloc(&device_b, size);
  cudaMalloc(&device_c, size);

  cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;
  printf("blocksPerGrid: %d\n", blocksPerGrid);

  addVec<<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, device_c, N);

  cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

  return host_c;
}