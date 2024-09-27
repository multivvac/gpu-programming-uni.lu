#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void genNFP(float* h, int n) {
  srand(time(0));
  for (int i = 0; i < n; i++) {
    h[i] = (float)rand() / RAND_MAX * 100.0f;
  }
}

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if  (i < n) {
    C[i] = A[i] + B[i];
  }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
  // Memory Allocation
  int size = n * sizeof(float);
  float* A_d;
  float* B_d;
  float* C_d;

  cudaMalloc((void**)&A_d, size);
  cudaMalloc((void**)&B_d, size);
  cudaMalloc((void**)&C_d, size);

  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  // call kernel code here
  vecAddKernel<<<ceil(n / 32.0), 32>>>(A_d, B_d, C_d, n);
  cudaFree(A_d);
  cudaFree(B_d);

  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
  cudaFree(C_d);
}

int main () {
  int n;
  printf("Please enter the size of matrix: ");
  scanf("%d", &n);

  int size = n * sizeof(float);

  float* A_h = (float *)malloc(size);
  float* B_h = (float *)malloc(size);
  float* C_h = (float *)malloc(size);


  genNFP(A_h, n);
  genNFP(B_h, n);

  vecAdd(A_h, B_h, C_h, n);

  free(A_h);
  free(B_h);
  free(C_h);
  return 0;
}