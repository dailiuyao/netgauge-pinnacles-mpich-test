#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:09:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A SR_APPFL
#PBS -k doe
#PBS -N ncclgauge_build
#PBS -o log/ncclgauge_build.out
#PBS -e log/ncclgauge_build.error

# Set environment variables

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda

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

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

export NCCL_GAUGE_HOME="/home/ldai8/ccl/netgauge-test/ncclguage"

# NCCL source location
NCCL_SRC_LOCATION="/home/ldai8/ccl/NCCL_profile"

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
            nvcc "$NVCC_GENCODE" -ccbin g++ -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" \
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
