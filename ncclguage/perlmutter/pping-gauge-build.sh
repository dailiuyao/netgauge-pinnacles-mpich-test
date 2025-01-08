#!/bin/bash
#SBATCH -A m4753 
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -t 00:09:99        # Run time (hh:mm:ss)
#SBATCH -J pping-build           # Job name
#SBATCH -o ./log/pping-build.o%j       # Name of stdout output file
#SBATCH -e ./log/pping-build.e%j       # Name of stderr error file
#SBATCH --gpu-bind=none

# Set environment variables

set -e

module load cudatoolkit
# module load cpe-cuda


export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3

# Update to include the correct path for MPI library paths
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export PATH=$CUDA_HOME/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include

# NCCL source location
NCCL_SRC_LOCATION="/global/homes/l/ldai8/ccl/NCCL_profile"

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

export NCCL_GAUGE_HOME="/global/homes/l/ldai8/ccl/netgauge-test/ncclguage"

export CRAY_ACCEL_TARGET=nvidia80

concurrency_sequence=(1 2 4 8 16)

experiments_number=5

for ((e = 0; e < experiments_number; e += 1)); do
    for i in "${concurrency_sequence[@]}"; do
        for mode in pping; do
            # Use proper variable expansion and quoting in the command
            CC -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" \
                -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lnccl -lcudart -lmpi \
                -D N_ITERS=${i} \
                "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.cc" -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_e_${e}.exe"

            # Verification of the output
            if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_e_${e}.exe" ]; then
                echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_e_${e}.exe"
            else
                echo "Compilation failed."
            fi
        done
    done
done
