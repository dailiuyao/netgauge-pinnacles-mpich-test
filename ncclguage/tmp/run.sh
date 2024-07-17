#!/bin/bash

# Set environment variables

module load mpich

source /home/liuyao/sbatch_sh/.nvccrc

export CUDA_HOME=/home/liuyao/software/cuda-11.6

# export MPI_HOME="/home/liuyao/software/mpich_4_1_1_pgcc"

export MPI_HOME="/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0"
export NCCL_SRC_LOCATION="/home/liuyao/scratch/deps/nccl"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"


$MPI_HOME/bin/mpirun -np 4 -ppn 2 -hosts node03,node04 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/ncclguage/tmp/ncclp2p.exe > output.log 2>&1

# $MPI_HOME/bin/mpirun -np 2 -ppn 2 -hosts node03 /home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/ncclguage/OneDevicePerThread.exe > output.log 2>&1