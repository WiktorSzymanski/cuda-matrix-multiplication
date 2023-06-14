#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void matrixMul(int *a, int *b, int *c, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int index = row * N + col;

  if (row < N && col < N) {
    int sum = 0;
    for (int i = 0; i < N; i++) {
      sum += a[row * N + i] * b[i * N + col];
    }
    c[index] = sum;
  }
}

void cpuMatrixMul(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c, int N) {
  int tmp;
  for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
      tmp = 0;
      for (int i = 0; i < N; i++) {
        tmp += a[row * N + i] * b[i * N + col];
      }
      c[row * N + col] = tmp;
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <N>" << std::endl;
    return 1;
  }
  int N = atoi(argv[1]);
  size_t size = N * N * sizeof(int);

  std::vector<int> a(N * N);
  std::vector<int> b(N * N);
  std::vector<int> c(N * N);
  std::vector<int> cCPU(N * N);

  std::generate(a.begin(), a.end(), []() { return rand() % 100; });
  std::generate(b.begin(), b.end(), []() { return rand() % 100; });

  int *cA, *cB, *cC;
  cudaMalloc(&cA, size);
  cudaMalloc(&cB, size);
  cudaMalloc(&cC, size);

  cudaMemcpy(cA, a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(cB, b.data(), size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 16;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  dim3 threads(threadsPerBlock, threadsPerBlock);
  dim3 blocks(blocksPerGrid, blocksPerGrid);

  auto start = std::chrono::high_resolution_clock::now();
  matrixMul<<<blocks, threads>>>(cA, cB, cC, N);
  cudaMemcpy(c.data(), cC, size, cudaMemcpyDeviceToHost);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "GPU time: " << duration.count() / 1000.0 << " ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  cpuMatrixMul(a, b, cCPU, N);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "CPU time: " << duration.count() / 1000.0 << " ms" << std::endl;

  if (c == cCPU) {
    printf("Success!\n");
  } else {
    printf("Failed!\n");
  }

  return 0;
}