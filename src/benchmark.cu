#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "benchmark.cuh"

/// CUDA kernel for benchmarking single-precision floating-point performance.
/// Each thread performs a fixed number of arithmetic operations (4 FLOPs per inner loop iteration)
/// to provide a consistent workload for performance measurement.
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

/// CUDA kernel for benchmarking double-precision floating-point performance.
/// Similar to the single-precision kernel, but uses double-precision types
/// and performs the same fixed number of arithmetic operations per thread.
__global__ void benchmarkKernel(double* out, const double* a, const double* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double sum = 0.0;
        for (int i = 0; i < 100000; ++i) {
            sum += a[idx] * b[idx] + a[idx] - b[idx];
        }
        out[idx] = sum;
    }
}

/// Measures the single-precision floating-point performance (GFLOPS) of the GPU.
/// This function sets up data on the host and device, launches the single-precision
/// benchmark kernel, uses CUDA events to accurately time its execution, and calculates
/// the GFLOPS based on the total operations and elapsed time.
///
/// @param N The number of elements processed by the kernel, influencing the total workload.
/// @return float The estimated GFLOPS.
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

    /// Use CUDA events for precise timing of kernel execution.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    benchmarkKernel <<<gridSize, blockSize>>> (d_out, d_a, d_b, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Wait for the kernel to finish

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    /// Calculate total floating-point operations and convert to GFLOPS.
    /// Total FLOPs = N * iterations_per_element * FLOPs_per_iteration
    float total_flops = N * 100000.0f * 4;
    float gflops = (total_flops / (milliseconds * 1e6));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return gflops;
}

/// Measures the double-precision floating-point performance (GDFLOPS) of the GPU.
/// This function is analogous to `measureGFLOPS` but uses double-precision types
/// and the corresponding kernel to measure double-precision performance.
///
/// @param N The number of elements processed by the kernel.
/// @return double The estimated GDFLOPS.
double measureGDFLOPS(int N) {
    double *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(double));
    cudaMalloc(&d_b, N * sizeof(double));
    cudaMalloc(&d_out, N * sizeof(double));
    double* h_a = new double[N];
    double* h_b = new double[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0;
        h_b[i] = 2.0;
    }
    cudaMemcpy(d_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_a;
    delete[] h_b;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    /// Use CUDA events for precise timing of kernel execution.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    benchmarkKernel<<<gridSize, blockSize>>>(d_out, d_a, d_b, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Wait for the kernel to finish
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    /// Calculate total double-precision floating-point operations and convert to GDFLOPS.
    /// Total FLOPs = N * iterations_per_element * FLOPs_per_iteration
    double total_Dflops = N * 100000.0 * 4;
    double gDflops = (total_Dflops / (milliseconds * 1e6));
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return gDflops;
}