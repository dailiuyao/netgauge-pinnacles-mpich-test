#!/bin/bash -l

set -e

module load cuda/11.3
module load intel  


export CUDA_HOME=/opt/apps/cuda/11.3

export NVCC_GENCODE="-gencode=arch=compute_75,code=sm_75"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Compilation command. Ensure to link against the MPI and NCCL libraries correctly.
nvcc $NVCC_GENCODE -ccbin g++ -L${CUDA_HOME}/lib64 -lcudart $1 -o ${1%.cu}.exe

# Verification of the output
if [ -f ${1%.cu}.exe ]; then
    echo "Compilation successful. Output file: ${1%.cu}.exe"
else
    echo "Compilation failed."
fi