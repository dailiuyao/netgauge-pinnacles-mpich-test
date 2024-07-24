#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:05:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A SR_APPFL 
#PBS -k doe
#PBS -N install-ofi-nccl
#PBS -o log/install-ofi-nccl.out
#PBS -e log/install-ofi-nccl.error

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

# Install and load libxml2 using Spack
spack load libxml2

export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda
export HWLOC_HOME=/home/ldai8/ccl/hwloc


# Explicitly set paths for libxml2 installed by Spack
export LD_LIBRARY_PATH=$(spack location -i libxml2)/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$(spack location -i libxml2)/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$(spack location -i libxml2)/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$(spack location -i libxml2)/include:$CPLUS_INCLUDE_PATH

export PATH=${MPI_HOME}/bin:$CUDA_HOME/bin:$PATH

export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

#install aws-ofi-nccl
pushd /home/ldai8/ccl/aws-ofi-nccl-1.7.4-aws

rm -rf build

mkdir build

cd build
../configure --prefix=/home/ldai8/ccl/aws-ofi-nccl-1.7.4-aws/build --with-libfabric=/opt/cray/libfabric/1.15.2.0/ --with-cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda --with-hwloc=$HWLOC_HOME     

# make clean

make -j8 && make install