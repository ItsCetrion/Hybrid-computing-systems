#ifdef __CUDACC__
// CUDA объявления
__global__ void addVec(float* a, float* b, float* c, int N);
#endif

int sumArray(const int* arr, int size);