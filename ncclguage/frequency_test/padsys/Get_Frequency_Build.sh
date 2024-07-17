#!/bin/bash -l

set -e

source /home/liuyao/sbatch_sh/.nvccrc

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# Compilation command. Ensure to link against the MPI and NCCL libraries correctly.
nvcc $NVCC_GENCODE -ccbin g++ -L${CUDA_HOME}/lib64 -lcudart $1 -o ${1%.cu}.exe

# Verification of the output
if [ -f ${1%.cu}.exe ]; then
    echo "Compilation successful. Output file: ${1%.cu}.exe"
else
    echo "Compilation failed."
fi