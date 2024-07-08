#!/bin/bash

# ---[ Script Setup ]---

set -e

# ---[ Set Up cuda/nccl/nccl-test/mpi ]---

spack load gcc@10.4.0 

spack load mpich@4.1.1

export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-10.4.0/mpich-4.1.1-j7lgvgtzrx6aj5k6a7lcs5xg4obnfi6i"

source /home/liuyao/sbatch_sh/.nvccrc

export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export NCCL_HOME="/home/liuyao/scratch/deps/nccl/build"
export C_INCLUDE_PATH="${NCCL_HOME}/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${NCCL_HOME}/include:$CPLUS_INCLUDE_PATH"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:$LD_LIBRARY_PATH"

export libnccl_HOME="/home/liuyao/software/netgauge_nccl/libnccl"

pushd ${libnccl_HOME}

nvcc -shared -o libMyNcclCode.so MyNcclCode.cu -I/${libnccl_HOME}  -I/${NCCL_HOME}/include -L/${NCCL_HOME}/lib -lnccl -L/${CUDA_HOME}/lib64 -lcudart -Xcompiler -fPIC

popd