#!/bin/bash


# Exit script on error
set -e 


### Load Modules (DIFFERENT DEPENDING ON SYSTEM) ###


# module load mpich/3.4.2-gcc-8.4.1
# export MPI_HOME="/opt/apps/mpi/mpich-3.4.2_gcc-8.4.1" 

# export MPI_HOME="/home/liuyao/software/mpich4_1_1"

# module load mpich

# export MPI_HOME="/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0"

# source /home/liuyao/sbatch_sh/.mpich_ucx

# export MPI_HOME="/home/liuyao/software/mpich4_1_1"

# export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
# export PATH=${MPI_HOME}/bin:$PATH
# export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

spack load gcc@10.4.0 

# spack load mpich@4.1.1 

# export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-10.4.0/mpich-4.1.1-j7lgvgtzrx6aj5k6a7lcs5xg4obnfi6i"

spack load openmpi@5.0.3

export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-8.5.0/openmpi-5.0.3-xsxjs6lg2gnrmhfygn5bpoyaeadarmcl"

export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

source /home/liuyao/sbatch_sh/.nvccrc

### Set environment variables ###

# Set target platform to reduce build time
# Note: A100 requires CUDA 11.0+, so I've set the default to '80'
NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
export NVCC_GENCODE

export DEPS_DIR_PADSYS="/home/liuyao/scratch"

# Set location to store NCCL source/repository
NCCL_SRC_LOCATION="${DEPS_DIR_PADSYS}/deps/NCCL_profile"
export NCCL_SRC_LOCATION

# export NCCL_COMMIT="primitive-time"

### Initial Information Collection ###

# Record Datetime, Hostname, and Working Directory
date;hostname;pwd

echo "### Running Tasks ###"
echo ""

# Get information about nvcc
echo "[INFO] NVCC info:"
NVCC_LOCATION="$(which nvcc)"
export NVCC_LOCATION
echo "${NVCC_LOCATION}"
nvcc --version
echo ""

# Set environment varaibles
echo "[DEBUG] CUDA_HOME has been set to: ${CUDA_HOME}"
echo "[DEBUG] MPI_HOME has been set to: ${MPI_HOME}"
echo ""


### NCCL Section ###

# Download NCCL
if [ ! -d "${NCCL_SRC_LOCATION}" ]; then
	echo "[INFO] Downloading NCCL repository..."
	git clone git@github.com:dailiuyao/NCCL_profile.git "${NCCL_SRC_LOCATION}"
elif [ -d "${NCCL_SRC_LOCATION}" ]; then 
	echo "[INFO] NCCL repository already exists."
fi
echo ""

# Enter NCCL dir
pushd "${NCCL_SRC_LOCATION}" || exit

# # Fetch latest changes
# git fetch --all

# # Checkout the correct commit
# git checkout "${NCCL_COMMIT}"

# Build NCCL

make clean

echo "[INFO] Building NCCL..."
make -j src.build CUDA_HOME=${CUDA_HOME} NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
echo ""

# Set environment variables that other tasks will use
echo "[INFO] Setting NCCL-related environment variables for other tasks..."
NCCL_HOME="${NCCL_SRC_LOCATION}/build" 
export NCCL_HOME
echo "[DEBUG] NCCL_HOME has been set to: ${NCCL_HOME}"

echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL!"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NCCL_HOME}/lib"
export LD_LIBRARY_PATH
PATH="${PATH}:${NCCL_HOME}/include"
export PATH
echo ""

# Exit NCCL dir
popd || exit
echo ""

