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

export GAUGE_D=0

for ((i = 1; i <= 1; i *= 8)); do
    for mode in pping; do
        # for sync_mode in sync group; do
        for sync_mode in sync; do
            if [ "${sync_mode}" == "sync" ]; then
                export D_SYNC=0
                export D_GROUP=0
            else
                export D_SYNC=0
                export D_GROUP=0
            fi

            # Use proper variable expansion and quoting in the command
            nvcc "$NVCC_GENCODE" -ccbin mpicc -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" \
                -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lnccl -lcudart -lmpi \
                -D N_ITERS=${i} \
                -D PROFILE_LYD_P2P_HOST_SYNC=${D_SYNC} \
                -D PROFILE_LYD_P2P_HOST_GROUP=${D_GROUP} \
                "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.cu" -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_${sync_mode}_d_${GAUGE_D}.exe"

            # Verification of the output
            if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_${sync_mode}_d_${GAUGE_D}.exe" ]; then
                echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_${sync_mode}_d_${GAUGE_D}.exe"
            else
                echo "Compilation failed."
            fi
        done
    done
done



# NCCL source location
NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile_D"

export GAUGE_D=2000

for ((i = 1; i <= 1; i *= 8)); do
    for mode in pping; do
        # for sync_mode in sync group; do
        for sync_mode in sync; do
            if [ "${sync_mode}" == "sync" ]; then
                export D_SYNC=0
                export D_GROUP=0
            else
                export D_SYNC=0
                export D_GROUP=0
            fi

            # Use proper variable expansion and quoting in the command
            nvcc "$NVCC_GENCODE" -ccbin mpicc -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" \
                -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lnccl -lcudart -lmpi \
                -D N_ITERS=${i} \
                -D PROFILE_LYD_P2P_HOST_SYNC=${D_SYNC} \
                -D PROFILE_LYD_P2P_HOST_GROUP=${D_GROUP} \
                "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.cu" -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_${sync_mode}_d_${GAUGE_D}.exe"

            # Verification of the output
            if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_${sync_mode}_d_${GAUGE_D}.exe" ]; then
                echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_${sync_mode}_d_${GAUGE_D}.exe"
            else
                echo "Compilation failed."
            fi
        done
    done
done