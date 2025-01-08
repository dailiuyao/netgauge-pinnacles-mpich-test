#!/bin/bash -l
#SBATCH -A m4753 
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH -J install-ofi-nccl           # Job name
#SBATCH -o ./log/install-ofi-nccl.o%j       # Name of stdout output file
#SBATCH -e ./log/install-ofi-nccl.e%j       # Name of stderr error file
#SBATCH --gpu-bind=none

module load cudatoolkit

export CRAY_ACCEL_TARGET=nvidia80

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3

export PATH=${MPI_HOME}/bin:$CUDA_HOME/bin:$PATH

export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

#install aws-ofi-nccl
pushd /global/homes/l/ldai8/ccl/aws-ofi-nccl-1.13.2-aws

rm -rf build

mkdir build

cd build
../configure --with-libfabric=/opt/cray/libfabric/1.20.1 --prefix=/global/homes/l/ldai8/ccl/aws-ofi-nccl-1.13.2-aws/build --with-cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2  

# make clean

make && make install