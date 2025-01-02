#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:09:59
#PBS -q debug-scaling
#PBS -l filesystems=home
#PBS -A SR_APPFL
#PBS -k doe
#PBS -N ncclgauge
#PBS -o log/test.out
#PBS -e log/test.error

# 1) Load necessary modules
module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

# 2) Load additional dependencies (libxml2 via spack)
spack load libxml2

# 3) Set MPI and CUDA environment variables
export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda

export PATH="${MPI_HOME}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

export C_INCLUDE_PATH="${MPI_HOME}/include:${CUDA_HOME}/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:$CPLUS_INCLUDE_PATH"

export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CUDNN_LIBRARY="${CUDA_HOME}/lib64"
export CUDNN_INCLUDE_DIR="${CUDA_HOME}/include"

export MPIEXEC_HOME=/opt/cray/pals/1.3.4

# 4) GPU architecture flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# 5) Set path to your NCCL gauge code and move there
export NCCL_GAUGE_HOME="/home/ldai8/ccl/netgauge-test/ncclguage"
cd "$NCCL_GAUGE_HOME/polaris"

# 6) Run the executable
"$NCCL_GAUGE_HOME/gauge/test.exe"