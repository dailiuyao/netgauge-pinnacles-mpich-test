#!/bin/bash


# Exit script on error
set -e 


### Load Modules (DIFFERENT DEPENDING ON SYSTEM) ###
module load cuda
# module load mpich  # [FIXME] Why does this cause NCCL build to error? Very strange...


### Set environment variables ###

# Set location of MPI (different depending on the system)
# MPI_HOME=/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0 
MPI_HOME=/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0
# MPI_HOME=/opt/spack/opt/spack/linux-rhel8-icelake/gcc-12.2.0/openmpi-4.1.4-o2fak4e2gbbavwtlfyedvsc6k2xibtai
export MPI_HOME
export PATH="${MPI_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:$LD_LIBRARY_PATH"

# Set location of CUDA on the machine
CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda"
# CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda"
export CUDA_HOME

# Set target platform to reduce build time
# Note: A100 requires CUDA 11.0+, so I've set the default to '80'
NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
export NVCC_GENCODE

# Set location to store NCCL source/repository
NCCL_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/nccl"
export NCCL_SRC_LOCATION
export NCCL_COMMIT="v2.17.1-1"

# Set location to store NCCL-PROFILE source/repository
NCCL_PROFILE_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/nccl_profile"
export NCCL_PROFILE_SRC_LOCATION
export NCCL_PROFILE_COMMIT="profile_steps"


# Set location to store MSCCL source/repository
MSCCL_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/msccl"
export MSCCL_SRC_LOCATION
export MSCCL_COMMIT="v0.7.4"

# Set location to store MSCCL_LYD source/repository
MSCCL_LYD_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/msccl_lyd"
export MSCCL_LYD_SRC_LOCATION
export MSCCL_LYD_COMMIT="algorithm_div_threads"

# Set location to store MSCCL Tools source/repository
MSCCLTOOLS_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/msccl-tools"
export MSCCLTOOLS_SRC_LOCATION

# Set location to store NCCL-Tests source/repository
NCCLTESTS_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/nccl-tests"
export NCCLTESTS_SRC_LOCATION

# Set location to store NCCL-Tests profile source/repository
NCCLTESTS_PROFILE_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/nccl-tests-profile"
export NCCLTESTS_PROFILE_SRC_LOCATION

# Set location to store NCCL-Tests-MSCCL source/repository
NCCLTESTS_MSCCL_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/nccl-tests-msccl"
export NCCLTESTS_MSCCL_SRC_LOCATION

# Set location to store NCCL-Tests-MSCCL-LYD source/repository
NCCLTESTS_MSCCL_LYD_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/nccl-tests-msccl-lyd"
export NCCLTESTS_MSCCL_LYD_SRC_LOCATION


# Set Python version to use (NOTE: You'll need to delete the python directory if you want to change the version later!)
PYTHON_SRC_VERSION="3.7.10"
export PYTHON_SRC_VERSION

# Set location to store Python source
PYTHON_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/pysrc"
export PYTHON_SRC_LOCATION

# Set location where Python should be installed after build
PYTHON_INSTALL_LOCATION="/home/ldai8/scratch/msccl_build/deps/python"
export PYTHON_INSTALL_LOCATION

# Set location to store cuDNN source
CUDNN_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/cudnn"
export CUDNN_SRC_LOCATION

# Set location where cuDNN should be installed after build
CUDNN_INSTALL_LOCATION="/home/ldai8/scratch/msccl_build/deps/cudnn/install"
export CUDNN_INSTALL_LOCATION

# Set location for the Python virtual environment to use when running tests
PY_VENV_LOCATION="/home/ldai8/scratch/msccl_build/venv"
export PY_VENV_LOCATION

# Set location to download Anaconda setup scripts
CONDA_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/conda_setup"
export CONDA_SRC_LOCATION

# Set location to install Anaconda
CONDA_INSTALL_LOCATION="/home/ldai8/scratch/msccl_build/deps/conda"
export CONDA_INSTALL_LOCATION

# Set string to use as the Anaconda environment name
CONDA_ENV_NAME="param_msccl"
export CONDA_ENV_NAME

# Set location to store PyTorch source/repository
PYTORCH_SRC_LOCATION="/home/ldai8/scratch/msccl_build/deps/pytorch"
export PYTORCH_SRC_LOCATION

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


# ### NCCL Section ###

# export PROFAPI=1
# # Download NCCL
# if [ ! -d "${NCCL_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading NCCL repository..."
# 	git clone https://github.com/nvidia/nccl.git "${NCCL_SRC_LOCATION}"
# elif [ -d "${NCCL_SRC_LOCATION}" ]; then 
# 	echo "[INFO] NCCL repository already exists."
# fi
# echo ""

# # Enter NCCL dir
# pushd "${NCCL_SRC_LOCATION}" || exit

# # Fetch latest changes
# git fetch --all

# # Checkout the correct commit
# git checkout "${NCCL_COMMIT}"

# # Build NCCL
# echo "[INFO] Building NCCL..."
# make clean
# make -j 16 src.build
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


# ### NCCL-PROFILE Section ###

# export PROFAPI=1
# # Download NCCL
# if [ ! -d "${NCCL_PROFILE_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading NCCL repository..."
# 	git clone git@github.com:dailiuyao/NCCL_profile.git "${NCCL_PROFILE_SRC_LOCATION}"
# elif [ -d "${NCCL_PROFILE_SRC_LOCATION}" ]; then 
# 	echo "[INFO] NCCL repository already exists."
# fi
# echo ""

# # Enter NCCL dir
# pushd "${NCCL_PROFILE_SRC_LOCATION}" || exit

# # # Fetch latest changes
# # git fetch --all

# # # Checkout the correct commit
# # git checkout "${NCCL_PROFILE_COMMIT}"

# # Build NCCL
# echo "[INFO] Building NCCL..."
# make clean
# make -j src.build
# echo ""

# # Set environment variables that other tasks will use
# echo "[INFO] Setting NCCL-related environment variables for other tasks..."
# NCCL_HOME="${NCCL_PROFILE_SRC_LOCATION}/build" 
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


# ### MSCCL Core Section ###

# # Download MSCCL
# if [ ! -d "${MSCCL_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading MSCCL repository..."
# 	git clone https://github.com/microsoft/msccl.git "${MSCCL_SRC_LOCATION}"
# elif [ -d "${MSCCL_SRC_LOCATION}" ]; then
# 	echo "[INFO] MSCCL repository already downloaded; will not re-download."
# fi
# echo ""

# # Enter MSCCL directory
# pushd "${MSCCL_SRC_LOCATION}" || exit

# # Fetch latest changes
# git fetch --all

# # Checkout the correct commit
# git checkout "${MSCCL_COMMIT}"

# # Build MSCCL
# echo "[INFO] Building MSCCL..."
# make clean
# make -j src.build
# echo ""

# # Create install package
# # [TODO]

# # Exist MSCCL directory
# popd || exit
# echo ""


# ### MSCCL_LYD Core Section ###

# # Download MSCCL_LYD
# if [ ! -d "${MSCCL_LYD_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading MSCCL_LYD repository..."
# 	git clone https://github.com/dailiuyao/msccl-lyd.git "${MSCCL_LYD_SRC_LOCATION}"
# elif [ -d "${MSCCL_LYD_SRC_LOCATION}" ]; then
# 	echo "[INFO] MSCCL_LYD repository already downloaded; will not re-download."
# fi
# echo ""

# # Enter MSCCL_LYD directory
# pushd "${MSCCL_LYD_SRC_LOCATION}" || exit

# # # Fetch latest changes
# # git fetch --all

# # # Checkout the correct commit
# # git checkout "${MSCCL_LYD_COMMIT}"

# # Build MSCCL_LYD
# echo "[INFO] Building MSCCL_LYD..."
# make clean
# make -j src.build
# echo ""

# # Create install package
# # [TODO]

# # Exist MSCCL_LYD directory
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
# make MPI=1 NCCL_HOME="${NCCL_SRC_LOCATION}/build"  

# # Exit NCCL Tests dir
# popd || exit
# echo ""

# ### NCCL Tests PROFILE Section ###

# # Download NCCL Tests
# if [ ! -d "${NCCLTESTS_PROFILE_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading NCCL Tests repository..."
# 	git clone https://github.com/nvidia/nccl-tests.git "${NCCLTESTS_PROFILE_SRC_LOCATION}"
# elif [ -d "${NCCLTESTS_PROFILE_SRC_LOCATION}" ]; then
# 	echo "[INFO] NCCL Tests PROFILE repository already exists."
# fi
# echo ""

# # Enter NCCL Tests dir
# pushd "${NCCLTESTS_PROFILE_SRC_LOCATION}" || exit
# echo ""
# make clean

# # Build NCCL Tests
# echo "[INFO] Building NCCL tests Profile"
# make MPI=1 MPI_HOME=${MPI_HOME} CUDA_HOME=${CUDA_HOME} NCCL_HOME="${NCCL_HOME}" -j  

# # Exit NCCL Tests dir
# popd || exit
# echo ""

# ### NCCL-Tests-MSCCL Section ###

# # Download NCCL-Tests-MSCCL
# if [ ! -d "${NCCLTESTS_MSCCL_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading NCCL Tests with MSCCL support repository..."
# 	git clone https://github.com/nvidia/nccl-tests.git "${NCCLTESTS_MSCCL_SRC_LOCATION}"
# elif [ -d "${NCCLTESTS_MSCCL_SRC_LOCATION}" ]; then
# 	echo "[INFO] NCCL Tests with MSCCL support repository already exists."
# fi
# echo ""

# # Enter NCCL-Tests-MSCCL dir
# pushd "${NCCLTESTS_MSCCL_SRC_LOCATION}" || exit
# echo ""

# # Build NCCL Tests with MSCCL support
# echo "[INFO] Building NCCL tests (nccl-tests) with MSCCL support..."
# make clean
# make MPI=1 NCCL_HOME="${MSCCL_SRC_LOCATION}/build/" -j  # Note: Use MSCCL's "version" of NCCL to build nccl-tests

# # Exit NCCL Tests dir
# popd || exit
# echo ""


### NCCL-Tests-MSCCL-LYD Section ###

# Download NCCL-Tests-MSCCL-LYD
if [ ! -d "${NCCLTESTS_MSCCL_LYD_SRC_LOCATION}" ]; then
	echo "[INFO] Downloading NCCL Tests with MSCCL support repository..."
	git clone https://github.com/nvidia/nccl-tests.git "${NCCLTESTS_MSCCL_LYD_SRC_LOCATION}"
elif [ -d "${NCCLTESTS_MSCCL_LYD_SRC_LOCATION}" ]; then
	echo "[INFO] NCCL Tests with MSCCL support repository already exists."
fi
echo ""

# Enter NCCL-Tests-MSCCL-LYD dir
pushd "${NCCLTESTS_MSCCL_LYD_SRC_LOCATION}" || exit
echo ""

# Build NCCL Tests with MSCCL support
echo "[INFO] Building NCCL tests (nccl-tests) with MSCCL support..."
make clean
make MPI=1 NCCL_HOME="${MSCCL_LYD_SRC_LOCATION}/build/" -j  # Note: Use MSCCL's "version" of NCCL to build nccl-tests

# Exit NCCL Tests dir
popd || exit
echo ""




### cuDNN Section ###

# # Download and prepare if necessary
# if [ ! -d "${CUDNN_SRC_LOCATION}" ]; then
# 	echo "[INFO] Did not find cuDNN src directory, creating..."
# 	mkdir -p "${CUDNN_SRC_LOCATION}"
# 	pushd "${CUDNN_SRC_LOCATION}" || exit
	
# 	if test -n "find '${CUDNN_SRC_LOCATION}' -maxdepth 1 -name 'cudnn-linux-*.tar.xz' -print -quit"; then
# 		echo "[FATAL] DID NOT FIND cuDNN SOURCE! CANNOT DOWNLOAD AUTOMATICALLY! YOU MUST UPLOAD MANUALLY TO ${CUDNN_SRC_LOCATION!}"
# 		exit 1
# 	fi
	
# 	echo "[INFO] Unarchiving cuDNN..."
# 	tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
	
# 	popd || exit
# elif [ -d "${CUDNN_SRC_LOCATION}" ]; then
# 	echo "[INFO] Found cuDNN src directory; will not re-download"
# fi

# # Install if necessary
# if [ ! -f "${CUDNN_INSTALL_LOCATION}/include" ]; then
# 	echo "[INFO] Did not find cuDNN install, will run install..."
	
# 	# Create install directories
# 	mkdir -p "${CUDNN_INSTALL_LOCATION}/include"
# 	mkdir -p "${CUDNN_INSTALL_LOCATION}/lib64"
	
# 	# Go to source directory to start copy
# 	pushd "${CUDNN_SRC_LOCATION}/cudnn-linux-x86_64-8.6.0.163_cuda11-archive" || exit
	
# 	# Run copy/install
# 	cp cudnn-*-archive/include/cudnn*.h "${CUDNN_INSTALL_LOCATION}/include"
# 	cp -P cudnn-*-archive/lib/libcudnn* "${CUDNN_INSTALL_LOCATION}/lib64"
	
# 	# Modify permissions
# 	chmod a+r /usr/local/cuda/include/cudnn*.h "${CUDNN_INSTALL_LOCATION}/lib64/libcudnn*"
	
# 	# Modify PATH
# 	# [TODO]
	
# 	popd || exit
# elif [ -f "${CUDNN_INSTALL_LOCATION}/include" ]; then
# 	echo "[INFO] Found cuDNN install; will not re-install"
# fi

# ### Python Section ###

# # Download and unzip if necessary
# if [ ! -d "${PYTHON_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading Python source code..."
# 	mkdir -p "${PYTHON_SRC_LOCATION}"
# 	pushd "${PYTHON_SRC_LOCATION}" || exit
# 	wget "https://www.python.org/ftp/python/${PYTHON_SRC_VERSION}/Python-${PYTHON_SRC_VERSION}.tgz"
# 	srun tar -xzf "Python-${PYTHON_SRC_VERSION}.tgz"
# 	cd "Python-${PYTHON_SRC_VERSION}"
# 	popd || exit	
# elif [ -d "${PYTHON_SRC_LOCATION}" ]; then
# 	echo "[INFO] Python source has already been downloaded; will not re-download."
# fi

# # Build and install if necessary
# if [ ! -f "${PYTHON_INSTALL_LOCATION}/bin/python3" ]; then
# 	echo "[INFO] Didn't find Python binary, will build and install..."
# 	pushd "${PYTHON_SRC_LOCATION}/Python-${PYTHON_SRC_VERSION}" || exit
# 	mkdir -p "${PYTHON_INSTALL_LOCATION}"
	
# 	echo "[INFO] Configuring Python compile..."
# 	srun ./configure --prefix="${PYTHON_INSTALL_LOCATION}" --enable-optimizations --with-ensurepip=install
	
# 	echo "[INFO] Building Python..."
# 	srun make -j
	
# 	echo "[INFO] Installing Python in ${PYTHON_INSTALL_LOCATION}..."
# 	srun make altinstall
	
# 	echo "[INFO] Linking python3 to version of python just installed..."
# 	ln -s "$(find $(realpath ${PYTHON_INSTALL_LOCATION}/bin) -regex '.*python.\..$')" "${PYTHON_INSTALL_LOCATION}/bin/python3"
	
# 	popd || exit
# elif [ -f "${PYTHON_INSTALL_LOCATION}/bin/python3" ]; then
# 	echo "[INFO] Found Python binary; will not re-build or re-install."
# fi

# # Update PATH to include installed Python version
# echo "[INFO] Updating PATH to include installed Python"
# PATH="${PYTHON_INSTALL_LOCATION}/bin:${PATH}"
# export PATH
# echo "[DEBUG] Value of PATH now: ${PATH}"

# # Print out information about python and pip
# echo "[DEBUG] Base Python version: $(python3 --version), location: $(which python3)"
# echo "[DEBUG] Base pip version: $(pip3 --version), location: $(which pip)"

# echo ""


# ### Anaconda Section ###

# # Download & Install Anaconda
# if [ ! -d "${CONDA_INSTALL_LOCATION}" ]; then
# 	echo "[INFO] Did not find existing Anaconda install, will download and install"

# 	# Create necessary directories
# 	mkdir -p "${CONDA_SRC_LOCATION}"
# 	mkdir -p "${CONDA_INSTALL_LOCATION}"
	
# 	# Go to setup directory
# 	pushd "${CONDA_SRC_LOCATION}" || exit
	
# 	# Download install script
# 	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	
# 	# Run install with specific install directory
# 	bash Miniconda3-latest-Linux-x86_64.sh -b -f -p "${CONDA_INSTALL_LOCATION}"
	
# 	# Exit install directory
# 	popd || exit
# else
# 	echo "[INFO] Found existing Anaconda install; will not reinstall."
# fi

# # Set up Anaconda to work with the shell
# source "${CONDA_INSTALL_LOCATION}/etc/profile.d/conda.sh" 

# # Update Anaconda
# conda upgrade -y conda

# # Create and activate an Anaconda environment
# conda create -y --name "${CONDA_ENV_NAME}" python="${PYTHON_SRC_VERSION}"
# conda activate "${CONDA_ENV_NAME}"


# ### MSCCL Tools ###

# # Clone the repository if it doesn't exist
# if [ ! -d "${MSCCLTOOLS_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading msccl-tools repo..."
# 	git clone -b algorithm-lyd https://github.com/dailiuyao/msccl-tools-lyd.git "${MSCCLTOOLS_SRC_LOCATION}"
# elif [ -d "${MSCCLTOOLS_SRC_LOCATION}" ]; then
# 	echo "[INFO] Found existing MSCCL Tools repository; will not re-download."
# fi
# echo ""

# # Create the virtual environment if necessary
# if [ ! -d "${PY_VENV_LOCATION}" ]; then
# 	echo "[INFO] Creating a Python virtual environment at ${PY_VENV_LOCATION}"
# 	python3 -m venv "${PY_VENV_LOCATION}"
# elif [ -d "$PY_VENV_LOCATION" ]; then
# 	echo "[INFO] Found existing Python virtual environment; will not create another."
# fi
# echo ""

# # Activate the virtual environment
# echo "[INFO] Activating virtual environment..."
# source "${PY_VENV_LOCATION}/bin/activate"

# # Print out infomration about python and pip
# echo "[DEBUG] Virtual environment Python version: $(python3 --version), location: $(which python3)"
# echo "[DEBUG] Virtual environment pip version: $(pip3 --version), location: $(which pip)"
# echo ""

# # Install msccl-tools with python
# echo "[INFO] Installing msccl-tools with virtual environment python..."
# # pip3 install git+https://github.com/microsoft/msccl.git
# # pip3 install "${MSCCL_SRC_LOCATION}"
# pip3 install "${MSCCLTOOLS_SRC_LOCATION}"
# echo ""

# ### PyTorch Section ###

# # Download PyTorch
# if [ ! -d "${PYTORCH_SRC_LOCATION}" ]; then
# 	echo "[INFO] Downloading PyTorch repository..."
# 	git clone --recursive https://github.com/pytorch/pytorch "${PYTORCH_SRC_LOCATION}"
# elif [ -d "${PYTORCH_SRC_LOCATION}" ]; then
# 	echo "[INFO] PyTorch repository already exists. Will update submodules..."
# 	pushd "${PYTORCH_SRC_LOCATION}" || exit
# 	git submodule sync
# 	git submodule update --init --recursive --jobs 0
# 	popd || exit
# fi
# echo ""

# # Install dependencies
# echo "[INFO] Installing PyTorch dependencies with conda..."
# conda install mkl mkl-include
# conda install -c pytorch magma-cuda110
# echo ""

# # Build PyTorch
# echo "[INFO] Building PyTorch from source..."
# pushd "${PYTORCH_SRC_LOCATION}" || exit
# export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# python setup.py install
# popd || exit
# echo ""

### End Tasks Section ###
echo ""

echo "### Done with tasks ###"

# ### User Info Section ###
# echo "To use the installed version of Python, run: export PATH=${PYTHON_INSTALL_LOCATION}/bin:\$PATH"

