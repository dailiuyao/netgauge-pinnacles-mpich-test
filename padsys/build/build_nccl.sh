#!/bin/bash

# ---[ Script Setup ]---

set -e

spack load gcc@10.4.0 

spack load openmpi@5.0.3

export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-10.4.0/openmpi-5.0.3-ltv5k5ckeuhvwzb2dnjqsb22eggfhmwh"

source /home/liuyao/sbatch_sh/.nvccrc

export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export NETGAUGE_HOME="/home/liuyao/software/netgauge_nccl"

export LD_LIBRARY_PATH="${NETGAUGE_HOME}/libnccl:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="${NETGAUGE_HOME}/libnccl:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${NETGAUGE_HOME}/libnccl:$CPLUS_INCLUDE_PATH"

export NCCL_HOME=/home/liuyao/scratch/deps/nccl/build
# export NCCL_TEST_HOME=/home/ldai8/NCCL/deps-nccl/nccl-tests/build
export C_INCLUDE_PATH="${NCCL_HOME}/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${NCCL_HOME}/include:$CPLUS_INCLUDE_PATH"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:$LD_LIBRARY_PATH"

export PATH=/home/liuyao/software/autoconf/bin:$PATH


echo "########## ENVIRONMENT ########"
echo "NCCL_LOCATION=${NCCL_HOME}"

cd ${NETGAUGE_HOME}

autoconf --version

# export CFLAGS=" -I${NETGAUGE_HOME}/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"
# export CPPFLAGS=" -I${NETGAUGE_HOME}/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"
# export CXXFLAGS=" -I${NETGAUGE_HOME}/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"


make clean

# ./configure --with-mpi=/home/liuyao/software/mpich4_1_1 \
#   LDFLAGS='-L${NETGAUGE_HOME} -L${NETGAUGE_HOME}/libnccl -lMyNcclCode -L/home/liuyao/NCCL/deps-nccl/nccl/build/lib -lnccl -L/home/liuyao/software/cuda-11.6/lib64 -lcudart -L/home/liuyao/software/mpich4_1_1/lib -lmpi -L//usr/lib64 -lpthread'

export LDFLAGS="-L${NETGAUGE_HOME} -L${NETGAUGE_HOME}/libnccl -L${NCCL_HOME}/lib -L${CUDA_HOME}/lib64 -L${MPI_HOME}/lib -L/usr/lib64"
export LIBS="-lMyNcclCode -lnccl -lcudart -lmpi -lpthread"
./configure --with-mpi=${MPI_HOME}
   


# sed -i 's/CXXFLAGS = -g -O2/CXXFLAGS = -g -O0/' Makefile

# sed -i 's/CFLAGS =/CFLAGS = -g -O0/' Makefile

# sed -i 's/CPPFLAGS =/CPPFLAGS = -g -O0/' Makefile 
  
# sed -i 's/CXXFLAGS = -g -O2/CXXFLAGS = -g -O0/' /home/liuyao/software/Netgauge/wnlib/Makefile 

make