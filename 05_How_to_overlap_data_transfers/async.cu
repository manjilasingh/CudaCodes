/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* 
 * NOTICE:
 * This code is not my original work.  
 * It is adapted from NVIDIA’s Developer Blog series 
 * “An Even Easier Introduction to CUDA” 
 * (https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
 * and their accompanying GitHub repository: 
 * https://github.com/NVIDIA-developer-blog/code-samples/tree/master/series/cuda-cpp
 *
 * All credit for the original implementation belongs to NVIDIA.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>

// Convenience function for checking CUDA runtime API results
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void kernel(float *a, size_t offset)
{
  size_t i = offset + threadIdx.x + blockIdx.x * (size_t)blockDim.x;
  float x = (float)i;
  float s = sinf(x);
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s * s + c * c);
}

float maxError(float *a, size_t n)
{
  float maxE = 0;
  for (size_t i = 0; i < n; i++) {
    float error = fabs(a[i] - 1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv)
{
  const size_t blockSize = 512;
  const size_t nStreams  = 8;
  const size_t n         = 2048ULL * 1024ULL * blockSize * nStreams;  // ~8.59e9 elements (~34 GB total)
  const size_t streamSize  = n / nStreams;
  const size_t streamBytes = streamSize * sizeof(float);
  const size_t bytes       = n * sizeof(float);
   
  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda(cudaSetDevice(devId));

  // allocate pinned host memory and device memory
  float *a, *d_a;
  checkCuda(cudaMallocHost((void**)&a, bytes));  // host pinned
  checkCuda(cudaMalloc((void**)&d_a, bytes));    // device

  float ms; // elapsed time in milliseconds
  
  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  checkCuda(cudaEventCreate(&dummyEvent));
  for (size_t i = 0; i < nStreams; ++i)
    checkCuda(cudaStreamCreate(&stream[i]));
  
  // baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
  kernel<<<n / blockSize, blockSize>>>(d_a, (size_t)0);
  checkCuda(cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 1: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (size_t i = 0; i < nStreams; ++i) {
    size_t offset = i * streamSize;
    checkCuda(cudaMemcpyAsync(&d_a[offset], &a[offset],
                              streamBytes, cudaMemcpyHostToDevice,
                              stream[i]));
    kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    checkCuda(cudaMemcpyAsync(&a[offset], &d_a[offset],
                              streamBytes, cudaMemcpyDeviceToHost,
                              stream[i]));
  }
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 2:
  // loop over copy, loop over kernel, loop over copy
  memset(a, 0, bytes);
  checkCuda(cudaEventRecord(startEvent, 0));
  for (size_t i = 0; i < nStreams; ++i) {
    size_t offset = i * streamSize;
    checkCuda(cudaMemcpyAsync(&d_a[offset], &a[offset],
                              streamBytes, cudaMemcpyHostToDevice,
                              stream[i]));
  }
  for (size_t i = 0; i < nStreams; ++i) {
    size_t offset = i * streamSize;
    kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  }
  for (size_t i = 0; i < nStreams; ++i) {
    size_t offset = i * streamSize;
    checkCuda(cudaMemcpyAsync(&a[offset], &d_a[offset],
                              streamBytes, cudaMemcpyDeviceToHost,
                              stream[i]));
  }
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // cleanup
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
  checkCuda(cudaEventDestroy(dummyEvent));
  for (size_t i = 0; i < nStreams; ++i)
    checkCuda(cudaStreamDestroy(stream[i]));
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}
