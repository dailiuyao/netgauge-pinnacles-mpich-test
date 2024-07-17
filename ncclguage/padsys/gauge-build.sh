#!/bin/bash

# Set environment variables

spack load gcc@10.4.0 

# spack load mpich@4.1.1 

# export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-10.4.0/mpich-4.1.1-j7lgvgtzrx6aj5k6a7lcs5xg4obnfi6i"

spack load openmpi@5.0.3

export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-10.4.0/openmpi-5.0.3-ltv5k5ckeuhvwzb2dnjqsb22eggfhmwh"

export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

source /home/liuyao/sbatch_sh/.nvccrc

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# NCCL source location
NCCL_SRC_LOCATION="/home/liuyao/scratch/deps/NCCL_profile"

NCCL_GAUGE_HOME="/home/liuyao/scratch/deps/netgauge-test/ncclguage"

GAUGE_D="0"

for ((i = 1; i <= 1; i *= 8)); do
    for mode in pping; do
        # for sync_mode in sync group; do
        for sync_mode in sync; do
            if [ "${sync_mode}" == "sync" ]; then
                D_SYNC="0"
                D_GROUP="0"
            else
                D_SYNC="0"
                D_GROUP="0"
            fi

            # Use proper variable expansion and quoting in the command
            nvcc "$NVCC_GENCODE" -ccbin mpicc -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" \
                -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lstdc++ -lnccl -lcudart -lmpi \
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
# NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile_D"

# GAUGE_D=2000

# for ((i = 1; i <= 1; i *= 8)); do
#     for mode in pping; do
#         # for sync_mode in sync group; do
#         for sync_mode in sync; do
#             if [ "${sync_mode}" == "sync" ]; then
#                 D_SYNC="0"
#                 D_GROUP="0"
#             else
#                 D_SYNC="0"
#                 D_GROUP="0"
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