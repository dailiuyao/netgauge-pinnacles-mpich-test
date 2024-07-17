#!/bin/bash

# Set environment variables

set -e

module load impi/19.0.5
module load cuda/11.3
module load intel  


export CUDA_HOME=/opt/apps/cuda/11.3
# export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64
export MPI_HOME=/opt/intel/compilers_and_libraries_2019.5.281/linux/mpi/intel64

export NVCC_GENCODE="-gencode=arch=compute_75,code=sm_75"

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
NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile"

export NCCL_GAUGE_HOME="/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/ncclguage"

for ((j = 0; j <= 1; j += 1)); do
    for ((i = 1; i <= 32; i *= 32)); do
        for mode in pping ; do
            # Use proper variable expansion and quoting in the command
            mpicc -I"${MPI_HOME}/include" \
                -L"${MPI_HOME}/lib" -lmpi \
                -D N_ITERS=${i} \
                -D GAUGE_D=${j} \
                "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_mpi.cc" -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_mpi_d_${j}_n_${i}.exe"

            # Verification of the output
            if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_mpi_d_${j}_n_${i}.exe" ]; then
                echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_mpi_d_${j}_n_${i}.exe"
            else
                echo "Compilation failed."
            fi
        done
    done
done