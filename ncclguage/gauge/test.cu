/***************************************************************************
 * File: clock64_example.cu
 * Compile: nvcc -arch=sm_70 clock64_example.cu -o clock64_example
 ***************************************************************************/

#include <cstdio>
#include <chrono>
#include <thread>
#include "cuda_runtime.h"

// Suppose your GPU runs at ~1410 MHz (1,410,000,000 cycles/second).
// Adjust for your actual GPU frequency.
#define GAUGE_GPU_FREQUENCY 1410  // in MHz

// A simple host function for a busy-wait of `ms` milliseconds
__host__ void busyWaitMilliseconds(int ms) {
    auto start = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::milliseconds(ms);
    while (std::chrono::high_resolution_clock::now() - start < duration) {
        // spin-wait
    }
}

// Kernel that records `clock64()` for thread 0 only
__global__ void clock64Kernel(unsigned long long* d_out) {
    // Get a per-thread clock64 timestamp
    unsigned long long timeVal = clock64();

    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Only store the timestamp if we are thread 0
    if (idx == 0) {
        d_out[0] = timeVal;
    }
}

int main() {
    // For simplicity, one block of 128 threads
    const int blockSize = 128;
    const int gridSize  = 1;

    // Device pointers where we'll store one timestamp each (for kernel #1 and #2).
    unsigned long long *d_out1 = nullptr, *d_out2 = nullptr;
    // Host copies of those timestamps
    unsigned long long h_out1 = 0, h_out2 = 0;

    // Allocate device memory for exactly one timestamp each
    cudaMalloc(&d_out1, sizeof(unsigned long long));
    cudaMalloc(&d_out2, sizeof(unsigned long long));

    // Launch first kernel, record a clock64() value in d_out1[0]
    clock64Kernel<<<gridSize, blockSize>>>(d_out1);
    cudaDeviceSynchronize();

    // Busy-wait on the CPU for 200 ms
    busyWaitMilliseconds(200);

    // Launch second kernel, record another clock64() value in d_out2[0]
    clock64Kernel<<<gridSize, blockSize>>>(d_out2);
    cudaDeviceSynchronize();

    // Copy the two timestamps back to host
    cudaMemcpy(&h_out1, d_out1, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_out2, d_out2, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Compute the difference in clock cycles
    unsigned long long diffCycles = (h_out2 > h_out1) ? (h_out2 - h_out1) : 0ull;

    // Convert clock cycles to milliseconds.
    // GAUGE_GPU_FREQUENCY is in MHz, so 1 cycle = 1 / (frequency*1e6) seconds.
    // diffCycles cycles * (1 / (freq*1e6)) seconds/cycle = diffCycles / (freq * 1e6) seconds
    // Multiply by 1e3 to get milliseconds.
    double diffMs = static_cast<double>(diffCycles) / (GAUGE_GPU_FREQUENCY * 1.0e6) * 1e3;

    printf("Kernel #1 vs #2 clock64 difference:\n");
    printf("  h_out1 = %llu cycles\n", (long long unsigned)h_out1);
    printf("  h_out2 = %llu cycles\n", (long long unsigned)h_out2);
    printf("  diffCycles = %llu\n", (long long unsigned)diffCycles);
    printf("  Approx time: %.3f ms (assuming %d MHz)\n", diffMs, GAUGE_GPU_FREQUENCY);

    // Cleanup
    cudaFree(d_out1);
    cudaFree(d_out2);

    return 0;
}