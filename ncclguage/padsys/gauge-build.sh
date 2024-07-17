#!/bin/bash

# Set environment variables

# module load mpich

# export MPI_HOME="/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0"

source /home/liuyao/sbatch_sh/.mpich_ucx

export MPI_HOME="/home/liuyao/software/mpich4_1_1"

# Update to include the correct path for MPI library paths
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

source /home/liuyao/sbatch_sh/.nvccrc

export CUDA_HOME=/home/liuyao/software/cuda-11.7
export PATH=/home/liuyao/software/cuda-11.7/bin:$PATH
export C_INCLUDE_PATH=/home/liuyao/software/cuda-11.7/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/liuyao/software/cuda-11.7/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/home/liuyao/software/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDACXX=/home/liuyao/software/cuda-11.7/bin/nvcc
export CUDNN_LIBRARY=/home/liuyao/software/cuda-11.7/lib64
export CUDNN_INCLUDE_DIR=/home/liuyao/software/cuda-11.7/include

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# NCCL source location
NCCL_SRC_LOCATION="/home/liuyao/scratch/deps/nccl"

export NCCL_GAUGE_HOME="/home/liuyao/scratch/deps/msccl_tools_lyd/examples/scripts/ncclguage"

for ((i = 1; i <= 32; i *= 2)); do
    for mode in pping ppong; do
        # Use proper variable expansion and quoting in the command
        nvcc "$NVCC_GENCODE" -ccbin g++ -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" \
            -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lnccl -lcudart -lmpi \
            -D N_ITERS=${i} \
            "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.cu" -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_${i}.exe"

        # Verification of the output
        if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_${i}.exe" ]; then
            echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_${i}.exe"
        else
            echo "Compilation failed."
        fi
    done
done