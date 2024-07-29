#!/bin/bash

# Set environment variables

set -e

module load cuda
module load openmpi

export MPI_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openmpi-4.1.6-lranp74
export CUDA_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc

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
NCCL_SRC_LOCATION="/u/ldai1/ccl-build/NCCL_profile"

export NCCL_GAUGE_HOME="/u/ldai1/ccl-build/netgauge-test/ncclguage"

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



# # NCCL source location
# NCCL_SRC_LOCATION="/u/ldai1/ccl-build/NCCL_profile_D"

# export GAUGE_D=2000

# for ((i = 1; i <= 1; i *= 8)); do
#     for mode in pping; do
#         # for sync_mode in sync group; do
#         for sync_mode in sync; do
#             if [ "${sync_mode}" == "sync" ]; then
#                 export D_SYNC=0
#                 export D_GROUP=0
#             else
#                 export D_SYNC=0
#                 export D_GROUP=0
#             fi

#             # Use proper variable expansion and quoting in the command
#             nvcc "$NVCC_GENCODE" -ccbin mpicc -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" \
#                 -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lnccl -lcudart -lmpi \
#                 -D N_ITERS=${i} \
#                 -D PROFILE_LYD_P2P_HOST_SYNC=${D_SYNC} \
#                 -D PROFILE_LYD_P2P_HOST_GROUP=${D_GROUP} \
#                 "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.cu" -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_${sync_mode}_d_${GAUGE_D}.exe"

#             # Verification of the output
#             if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_${sync_mode}_d_${GAUGE_D}.exe" ]; then
#                 echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_${sync_mode}_d_${GAUGE_D}.exe"
#             else
#                 echo "Compilation failed."
#             fi
#         done
#     done
# done