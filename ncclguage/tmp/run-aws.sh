#!/bin/bash

# Set environment variables
export CUDA_HOME=/usr/local/cuda

export MPI_HOME="/opt/amazon/openmpi"

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

# NCCL source location
NCCL_SRC_LOCATION="/opt/nccl"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

$MPI_HOME/bin/mpirun -np 4 -ppn 2 -hosts node03,node04 /mnt/sharedfs/ly-experiments/msccl_tools_lyd/examples/scripts/ncclguage/tmp/ncclp2p.exe > output.log 2>&1