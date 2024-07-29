#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaError_t status;
    int device;
    int clockRate;

    // Get the current device
    status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        printf("cudaGetDevice failed: %s\n", cudaGetErrorString(status));
        return 1;
    }

    // Get the clock rate of the current device
    status = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, device);
    if (status != cudaSuccess) {
        printf("cudaDeviceGetAttribute failed: %s\n", cudaGetErrorString(status));
        return 1;
    }

    // Print the clock rate in kHz (note: to convert to Hz, multiply by 1000)
    printf("GPU Clock Rate: %d kHz\n", clockRate);

    // Optionally, convert to Hz for calculations
    float clockRateHz = clockRate * 1000.0f;
    printf("GPU Clock Rate: %.0f Hz\n", clockRateHz);

    return 0;
}
