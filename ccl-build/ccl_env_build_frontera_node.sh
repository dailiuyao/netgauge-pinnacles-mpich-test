#!/bin/bash



set -e

module load gcc/9.1.0
module load impi/19.0.5
module load cuda/11.3


export CUDA_HOME=/opt/apps/cuda/11.3
# export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64
export MPI_HOME=/opt/intel/compilers_and_libraries_2019.5.281/linux/mpi/intel64

export NVCC_GENCODE="-gencode=arch=compute_75,code=sm_75"

# ################################## NCCL ########################################

# Set location to store NCCL source/repository
NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl"
export NCCL_SRC_LOCATION
export NCCL_COMMIT="v2.17.1-1"

# Set location to store NCCL_TEST source/repository
NCCLTESTS_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests"
export NCCLTESTS_SRC_LOCATION

# Download NCCL
if [ ! -d "${NCCL_SRC_LOCATION}" ]; then
      echo "[INFO] Downloading NCCL repository..."
      git clone https://github.com/NVIDIA/nccl.git "${NCCL_SRC_LOCATION}"
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
echo "[INFO] Building NCCL..."
make clean
make -j src.build
echo ""

# Exit NCCL dir
popd || exit
echo ""


# ### NCCL Tests Section ###

# echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL!"

# NCCL_HOME="${NCCL_SRC_LOCATION}/build" 
# export NCCL_HOME
# echo "[DEBUG] NCCL_HOME has been set to: ${NCCL_HOME}"

# LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH
# PATH="${NCCL_HOME}/include:${PATH}"
# export PATH
# echo ""

# # Download NCCL Tests
# if [ ! -d "${NCCLTESTS_SRC_LOCATION}" ]; then
#       echo "[INFO] Downloading NCCL Tests repository..."
#       git clone https://github.com/nvidia/nccl-tests.git "${NCCLTESTS_SRC_LOCATION}"
# elif [ -d "${NCCLTESTS_SRC_LOCATION}" ]; then
#       echo "[INFO] NCCL Tests repository already exists."
# fi
# echo ""

# # Enter NCCL Tests dir
# pushd "${NCCLTESTS_SRC_LOCATION}" || exit
# echo ""
# make clean

# # Build NCCL Tests
# echo "[INFO] Building NCCL tests (nccl-tests)"
# make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME=${NCCL_HOME}  

# # Exit NCCL Tests dir
# popd || exit
# echo ""

# # ################################## MSCCL PROFIEL ########################################

# # Set location to store NCCL source/repository
# MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/msccl-lyd"
# export MSCCL_SRC_LOCATION
# export MSCCL_COMMIT="algorithm_test_CCLadviser"

# # Set location to store NCCL_TEST source/repository
# NCCLTESTS_MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile-msccl"
# export NCCLTESTS_MSCCL_SRC_LOCATION
# export NCCLTESTS_MSCCL_COMMIT="nccl-test-profile-msccl-frontera"

# ### MSCCL-Section ###
# # Download MSCCL
# if [ ! -d "${MSCCL_SRC_LOCATION}" ]; then
#       echo "[INFO] Downloading MSCCL repository..."
#       git clone git@github.com:dailiuyao/msccl-lyd.git "${MSCCL_SRC_LOCATION}"
# elif [ -d "${MSCCL_SRC_LOCATION}" ]; then 
#       echo "[INFO] MSCCL repository already exists."
# fi
# echo ""

# # Enter MSCCL dir
# pushd "${MSCCL_SRC_LOCATION}" || exit

# # # Fetch latest changes
# # git fetch --all

# # # Checkout the correct commit
# # git checkout "${MSCCL_COMMIT}"

# # Build MSCCL
# echo "[INFO] Building MSCCL..."
# make clean > /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/output.log 2>&1
# make -j src.build > /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/output.log 2>&1
# echo ""

# # Exit MSCCL dir
# popd || exit
# echo ""


# ### NCCL Tests MSCCL Section ###

# echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include MSCCL!"

# # Set location to store NCCL source/repository
# MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/msccl-lyd"
# export MSCCL_SRC_LOCATION

# # Set location to store NCCL_TEST source/repository
# NCCLTESTS_MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile-msccl"
# export NCCLTESTS_MSCCL_SRC_LOCATION

# MSCCL_HOME="${MSCCL_SRC_LOCATION}/build" 
# export MSCCL_HOME
# echo "[DEBUG] NCCL_HOME has been set to: ${MSCCL_HOME}"

# LD_LIBRARY_PATH="${MSCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH
# PATH="${MSCCL_HOME}/include:${PATH}"
# export PATH
# echo ""

# # Download NCCL Tests with MSCCL
# if [ ! -d "${NCCLTESTS_MSCCL_SRC_LOCATION}" ]; then
#       echo "[INFO] Downloading NCCL Tests with MSCCL repository..."
#       git clone git@github.com:dailiuyao/nccl-tests.git "${NCCLTESTS_MSCCL_SRC_LOCATION}"
# elif [ -d "${NCCLTESTS_MSCCL_SRC_LOCATION}" ]; then
#       echo "[INFO] NCCL Tests with MSCCL repository already exists."
# fi
# echo ""


# NCCLTESTS_MSCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile-msccl"

# # Enter NCCL Tests with MSCCL dir
# pushd "${NCCLTESTS_MSCCL_SRC_LOCATION}" || exit
# echo ""

# # # Fetch latest changes
# # git fetch --all

# # # Checkout the correct commit
# # git checkout "${NCCLTESTS_MSCCL_COMMIT}"

# make clean

# # Build NCCL Tests with MSCCL
# echo "[INFO] Building NCCL tests (MSCCL)"
# make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME=${MSCCL_HOME}  

# # Exit NCCL Tests with MSCCL dir
# popd || exit
# echo ""

# # # ################################## NCCL_PROFILE ########################################

# # Set location to store NCCL source/repository
# NCCL_PROFILE_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile"
# export NCCL_PROFILE_SRC_LOCATION
# # export NCCL_PROFILE_COMMIT="profile_steps"

# # Set location to store NCCL_TEST source/repository
# NCCLTESTS_NCCL_PROFILE_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile"
# export NCCLTESTS_NCCL_PROFILE_SRC_LOCATION
# # export NCCLTESTS_NCCL_PROFILE_COMMIT="nccl-test-profile"

# ### NCCL_PROFILE-Section ###
# # Download NCCL_PROFILE
# if [ ! -d "${NCCL_PROFILE_SRC_LOCATION}" ]; then
#       echo "[INFO] Downloading NCCL_PROFILE repository..."
#       git clone git@github.com:dailiuyao/NCCL_profile.git "${NCCL_PROFILE_SRC_LOCATION}"
# elif [ -d "${NCCL_PROFILE_SRC_LOCATION}" ]; then 
#       echo "[INFO] NCCL_PROFILE repository already exists."
# fi
# echo ""

# # Enter NCCL_PROFILE dir
# pushd "${NCCL_PROFILE_SRC_LOCATION}" || exit

# # # Fetch latest changes
# # git fetch --all

# # # Checkout the correct commit
# # git checkout "${NCCL_PROFILE_COMMIT}"

# # Build NCCL_PROFILE
# echo "[INFO] Building NCCL_PROFILE..."
# make clean
# make -j src.build
# echo ""

# # Exit NCCL_PROFILE dir
# popd || exit
# echo ""


# ### NCCL Tests NCCL_PROFILE Section ###

# export DEBUG=0

# echo "[INFO] Updating LD_LIBRARY_PATH and PATH to include NCCL_PROFILE!"

# NCCL_PROFILE_HOME="${NCCL_PROFILE_SRC_LOCATION}/build" 
# export NCCL_PROFILE_HOME
# echo "[DEBUG] NCCL_HOME has been set to: ${NCCL_PROFILE_HOME}"

# LD_LIBRARY_PATH="${NCCL_PROFILE_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH
# PATH="${NCCL_PROFILE_HOME}/include:${PATH}"
# export PATH
# echo ""

# # Download NCCL Tests with NCCL_PROFILE
# if [ ! -d "${NCCLTESTS_NCCL_PROFILE_SRC_LOCATION}" ]; then
#       echo "[INFO] Downloading NCCL Tests with NCCL_PROFILE repository..."
#       git clone git@github.com:dailiuyao/nccl-tests.git "${NCCLTESTS_NCCL_PROFILE_SRC_LOCATION}"
# elif [ -d "${NCCLTESTS_NCCL_PROFILE_SRC_LOCATION}" ]; then
#       echo "[INFO] NCCL Tests with NCCL_PROFILE repository already exists."
# fi
# echo ""


# NCCLTESTS_NCCL_PROFILE_SRC_LOCATION="/home1/09168/ldai1/ccl-build/nccl-tests-profile"

# # Enter NCCL Tests with NCCL_PROFILE dir
# pushd "${NCCLTESTS_NCCL_PROFILE_SRC_LOCATION}" || exit
# echo ""

# # # Fetch latest changes
# # git fetch --all

# # # Checkout the correct commit
# # git checkout "${NCCLTESTS_NCCL_PROFILE_COMMIT}"


# make clean

# # Build NCCL Tests with NCCL_PROFILE
# echo "[INFO] Building NCCL tests (NCCL_PROFILE)"
# make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME=${NCCL_PROFILE_HOME} > /home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/output.log 2>&1  

# # Exit NCCL Tests with NCCL_PROFILE dir
# popd || exit
# echo ""