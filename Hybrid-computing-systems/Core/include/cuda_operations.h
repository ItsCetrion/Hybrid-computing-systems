#ifndef CUDA_OPERATIONS_H
#define CUDA_OPERATIONS_H

#include <stdlib.h>

// Функция подготовки памяти GPU и запуска ядра с заданной конфигурацией числа блоков и потоков в них
void addVectorsExtended(const float *h_a, const float *h_b, float *h_result, const int n, const int blockSize, const int gridSize);

// Функция подготовки памяти GPU и запуска ядра в оптимальной конфигурации числа блоков и потоков
void addVectorsOptimal(const float *h_a, const float *h_b, float *h_result, const int n);

// Функция запуска ядра с уже подготовленной ранее памятью GPU
void addVectorsKernelRun(const float *d_a, const float *d_b, float *d_result, const int n, const int blockSize, const int gridSize, float *kernel_ms);


// Функции-обёртки над CUDA API для вызова из чистого C++
void* cuda_malloc(std::size_t size);
void cuda_free(void *ptr);
void cuda_memcpy_host_to_device(void *dst, const void *src, std::size_t size);
void cuda_memcpy_device_to_host(void *dst, const void *src, std::size_t size);


#endif // CUDA_OPERATIONS_H

