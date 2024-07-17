#!/bin/bash

#SBATCH -J nccl-gauge-build           # Job name
#SBATCH -o ./nccl-gauge-build.o%j       # Name of stdout output file
#SBATCH -e ./nccl-gauge-build.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH --exclude=c197-072
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A nccl-gauge-build       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

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

export mode="allreduce"

# Use proper variable expansion and quoting in the command
nvcc "$NVCC_GENCODE" -ccbin mpicc -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" \
    -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lnccl -lcudart -lmpi \
    "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.cu" -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.exe"

# Verification of the output
if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.exe" ]; then
    echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.exe"
else
    echo "Compilation failed."
fi