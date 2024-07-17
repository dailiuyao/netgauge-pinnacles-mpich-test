#!/bin/bash -l

# Set environment variables
module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"


./GetCudaFrequency.exe