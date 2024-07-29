#!/bin/bash -l

set -e

module load cuda

export CUDA_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}


./GetCudaFrequency.exe


# GPU Clock Rate: 1410000 kHz
# GPU Clock Rate: 1410000000 Hz