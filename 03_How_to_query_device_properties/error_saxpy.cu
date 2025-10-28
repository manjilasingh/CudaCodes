#include <iostream>
#include <math.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Intentional out-of-bounds access
  y[i + n * 100] = a*x[i] + y[i];
}

int main(void)
{
  int N = 20 * (1 << 20);
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start);

  // Invalid configuration - too many threads
  saxpy<<<(N+255)/256, 8256>>>(N, 2.0, d_x, d_y);
cudaError_t errSync  = cudaGetLastError();
cudaError_t errAsync = cudaDeviceSynchronize();
if (errSync != cudaSuccess) 
  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
if (errAsync != cudaSuccess)
  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

  
  cudaEventRecord(stop);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    maxError = max(maxError, abs(y[i]-4.0f));
  }

  printf("Max error: %f\n", maxError);
}