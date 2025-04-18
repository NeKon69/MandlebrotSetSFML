#include <cuda_runtime.h>


// CUDA kernel benchmark
__global__ void benchmarkKernel(float* out, const float* a, const float* b, int N);

// Function to measure GFLOPS with given N
float measureGFLOPS(int N);