#!/bin/bash
#SBATCH -A m4753 
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH -J ccl-env-build           # Job name
#SBATCH -o ./build-log/ccl-env-build.o%j       # Name of stdout output file
#SBATCH -e ./build-log/ccl-env-build.e%j       # Name of stderr error file
#SBATCH --gpu-bind=none


set -e

module load cudatoolkit

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

# # ################################## NCCL_PROFILE ########################################

# Set location to store NCCL source/repository
NCCL_PROFILE_SRC_LOCATION="/global/homes/l/ldai8/ccl/NCCL_profile"
export NCCL_PROFILE_SRC_LOCATION
# export NCCL_PROFILE_COMMIT="profile_steps"

### NCCL_PROFILE-Section ###
# Download NCCL_PROFILE
if [ ! -d "${NCCL_PROFILE_SRC_LOCATION}" ]; then
      echo "[INFO] Downloading NCCL_PROFILE repository..."
      git clone git@github.com:dailiuyao/NCCL_profile.git "${NCCL_PROFILE_SRC_LOCATION}"
elif [ -d "${NCCL_PROFILE_SRC_LOCATION}" ]; then 
      echo "[INFO] NCCL_PROFILE repository already exists."
fi
echo ""

# Enter NCCL_PROFILE dir
pushd "${NCCL_PROFILE_SRC_LOCATION}" || exit

# # Fetch latest changes
# git fetch --all

# # Checkout the correct commit
# git checkout "${NCCL_PROFILE_COMMIT}"

# Build NCCL_PROFILE
echo "[INFO] Building NCCL_PROFILE..."
make clean
make -j src.build
echo ""

# Exit NCCL_PROFILE dir
popd || exit
echo ""

