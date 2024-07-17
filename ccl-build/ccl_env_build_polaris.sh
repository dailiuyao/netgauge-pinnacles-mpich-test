#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:09:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A MPICH_MCS
#PBS -k doe
#PBS -N ccl-build
#PBS -o build-log/ccl-build.out
#PBS -e build-log/ccl-build.error

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda

# Set location to store MSCCL source/repository
MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/msccl-lyd-buff1"
export MSCCL_SRC_LOCATION
export MSCCL_COMMIT="algorithm_test_CCLadviser"

export MSCCL_HOME=${MSCCL_SRC_LOCATION}/build

# Set location to store NCCL_TEST_PROFILE source/repository
NCCLTESTS_PROFILE_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl-tests-profile"
export NCCLTESTS_PROFILE_SRC_LOCATION
export NCCL_TEST_PROFILE_COMMIT="nccl-test-profile-nccl"

# Set location to store NCCL-Tests-MSCCL-LYD source/repository
NCCLTESTS_MSCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl-tests-msccl"
export NCCLTESTS_MSCCL_SRC_LOCATION
export NCCL_TEST_PROFILE_MSCCL_COMMIT="nccl-test-profile-msccl"

# Set location to store NCCL-PROFILE source/repository
NCCL_PROFILE_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl_profile"
export NCCL_PROFILE_SRC_LOCATION
export NCCL_PROFILE_COMMIT="primitive-time"

# Set location to store NCCL-PROFILE source/repository
NCCL_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl"
export NCCL_SRC_LOCATION
export NCCL_COMMIT="v2.12.12-1"

# Set location to store NCCL_TEST source/repository
NCCLTESTS_SRC_LOCATION="/home/yuke/ncclPG/CCL-LYD/nccl-tests"
export NCCLTESTS_SRC_LOCATION

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

### NCCL-PROFILE Section ###

export PROFAPI=1
# Download NCCL
if [ ! -d "${NCCL_PROFILE_SRC_LOCATION}" ]; then
	echo "[INFO] Downloading NCCL repository..."
	git clone git@github.com:dailiuyao/NCCL_profile.git "${NCCL_PROFILE_SRC_LOCATION}"
elif [ -d "${NCCL_PROFILE_SRC_LOCATION}" ]; then 
	echo "[INFO] NCCL repository already exists."
fi
echo ""

# Enter NCCL dir
pushd "${NCCL_PROFILE_SRC_LOCATION}" || exit

# # Fetch latest changes
# git fetch --all

# # Checkout the correct commit
# git checkout "${NCCL_PROFILE_COMMIT}"

# Build NCCL
echo "[INFO] Building NCCL_PROFILE..."
make clean
make -j src.build
echo ""

# Set environment variables that other tasks will use
echo "[INFO] Setting NCCL-related environment variables for other tasks..."
NCCL_HOME="${NCCL_PROFILE_SRC_LOCATION}/build" 
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


# ### NCCL-Tests-PROFILE Section ###

# # Download NCCL-Tests-PROFILE
# if [ ! -d "${NCCLTESTS_PROFILE_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading NCCL Tests with profile support repository..."
# 	git clone git@github.com:dailiuyao/nccl-tests.git "${NCCLTESTS_PROFILE_SRC_LOCATION}"
# elif [ -d "${NCCLTESTS_PROFILE_SRC_LOCATION}" ]; then
# 	echo "[INFO] NCCL Tests with PROFILE support repository already exists."
# fi
# echo ""

# # Enter NCCL-Tests-MSCCL-TEST dir
# pushd "${NCCLTESTS_PROFILE_SRC_LOCATION}" || exit
# echo ""

# # # Fetch latest changes
# # git fetch --all

# # # Checkout the correct commit
# # git checkout "${NCCL_TEST_PROFILE_COMMIT}"


# # Build NCCL Tests with MSCCL support
# echo "[INFO] Building NCCL tests (nccl-tests) with PROFILE support..."
# make clean
# make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME="${NCCL_HOME}" -j

# # Exit NCCL Tests dir
# popd || exit
# echo ""


# ### MSCCL Core Section ###

# # rm -rf "${MSCCL_SRC_LOCATION}" 

# # Download MSCCL
# if [ ! -d "${MSCCL_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading MSCCL repository..."
# 	git clone https://github.com/dailiuyao/msccl-lyd.git "${MSCCL_SRC_LOCATION}"
# elif [ -d "${MSCCL_SRC_LOCATION}" ]; then
# 	echo "[INFO] MSCCL repository already downloaded; will not re-download."
# fi
# echo ""

# # Enter MSCCL directory
# pushd "${MSCCL_SRC_LOCATION}" || exit

# ## Fetch latest changes
# #git fetch --all

# ## Checkout the correct commit
# #git checkout "${MSCCL_COMMIT}"

# # Build MSCCL
# echo "[INFO] Building MSCCL..."
# make -j src.build
# echo ""

# # Create install package
# # [TODO]

# # Exist MSCCL directory
# popd || exit
# echo ""


#### NCCL-Tests-MSCCL-LYD Section ###
#
## rm -rf "${NCCLTESTS_MSCCL_SRC_LOCATION}" 
#
## Download NCCL-Tests-MSCCL-LYD
#if [ ! -d "${NCCLTESTS_MSCCL_SRC_LOCATION}" ]; then
#	echo "[INFO] Downloading NCCL Tests with MSCCL support repository..."
#	git clone https://github.com/dailiuyao/nccl-tests.git "${NCCLTESTS_MSCCL_SRC_LOCATION}"
#elif [ -d "${NCCLTESTS_MSCCL_SRC_LOCATION}" ]; then
#	echo "[INFO] NCCL Tests with MSCCL support repository already exists."
#fi
#echo ""
#
## Enter NCCL-Tests-MSCCL-TEST dir
#pushd "${NCCLTESTS_MSCCL_SRC_LOCATION}" || exit
#echo ""
#
### Fetch latest changes
##git fetch --all
#
### Checkout the correct commit
##git checkout "${NCCL_TEST_PROFILE_MSCCL_COMMIT}"
#
## Build NCCL Tests with MSCCL support
#echo "[INFO] Building NCCL tests (nccl-tests) with MSCCL support..."
#make clean
#make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME="${MSCCL_HOME}" -j  # Note: Use MSCCL's "version" of NCCL to build nccl-tests
#
## Exit NCCL Tests dir
#popd || exit
#echo ""

# ### NCCL-Section ###

# export PROFAPI=1
# # Download NCCL
# if [ ! -d "${NCCL_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading NCCL repository..."
# 	git clone https://github.com/NVIDIA/nccl.git "${NCCL_SRC_LOCATION}"
# elif [ -d "${NCCL_SRC_LOCATION}" ]; then 
# 	echo "[INFO] NCCL repository already exists."
# fi
# echo ""

# # Enter NCCL dir
# pushd "${NCCL_SRC_LOCATION}" || exit

# # # Fetch latest changes
# # git fetch --all

# # # Checkout the correct commit
# # git checkout "${NCCL_COMMIT}"

# # Build NCCL
# echo "[INFO] Building NCCL..."
# make clean
# make -j src.build
# echo ""

# # Set environment variables that other tasks will use
# echo "[INFO] Setting NCCL-related environment variables for other tasks..."
# NCCL_HOME="${NCCL_SRC_LOCATION}/build" 
# export NCCL_HOME
# echo "[DEBUG] NCCL_HOME has been set to: ${NCCL_HOME}"

# echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL!"
# LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NCCL_HOME}/lib"
# export LD_LIBRARY_PATH
# PATH="${PATH}:${NCCL_HOME}/include"
# export PATH
# echo ""

# # Exit NCCL dir
# popd || exit
# echo ""


# ### NCCL Tests Section ###

# # Download NCCL Tests
# if [ ! -d "${NCCLTESTS_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading NCCL Tests repository..."
# 	git clone https://github.com/nvidia/nccl-tests.git "${NCCLTESTS_SRC_LOCATION}"
# elif [ -d "${NCCLTESTS_SRC_LOCATION}" ]; then
# 	echo "[INFO] NCCL Tests repository already exists."
# fi
# echo ""

# # Enter NCCL Tests dir
# pushd "${NCCLTESTS_SRC_LOCATION}" || exit
# echo ""
# make clean

# # Build NCCL Tests
# echo "[INFO] Building NCCL tests (nccl-tests)"
# make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME="${NCCL_SRC_LOCATION}/build"  


# # make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME="${MSCCL_HOME}"

# # Exit NCCL Tests dir
# popd || exit
# echo ""