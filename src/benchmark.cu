#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "benchmark.cuh"

__global__ void benchmarkKernel(float* out, const float* a, const float* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < 100000; ++i) {
            sum += a[idx] * b[idx] + a[idx] - b[idx];
        }
        out[idx] = sum;
    }
}

float measureGFLOPS(int N) {
    float* d_a, * d_b, * d_out;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    float* h_a = new float[N];
    float* h_b = new float[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    delete[] h_a;
    delete[] h_b;

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    benchmarkKernel << <gridSize, blockSize >> > (d_out, d_a, d_b, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float total_flops = N * 100000.0f * 4;
    float gflops = (total_flops / (milliseconds * 1e6));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return gflops;
}