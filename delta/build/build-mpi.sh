#!/bin/bash


# ---[ Script Setup ]---

set -e


module load cuda
module load openmpi

export MPI_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openmpi-4.1.6-lranp74
export CUDA_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc

export LD_LIBRARY_PATH="${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
export PATH="${MPI_HOME}/bin:${CUDA_HOME}/bin:$PATH"
export C_INCLUDE_PATH="${MPI_HOME}/include:${CUDA_HOME}/include:$C_INCLUDE_PATH"


echo "########## ENVIRONMENT ########"
echo "NCCL_LOCATION=${NCCL_HOME}"

export NETGAUGE_HOME="/u/ldai1/ccl-build/netgauge_default"

cd ${NETGAUGE_HOME} 

autoconf --version

# export CFLAGS=" -I/home/liuyao/software/Netgauge/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"
# export CPPFLAGS=" -I/home/liuyao/software/Netgauge/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"
# export CXXFLAGS=" -I/home/liuyao/software/Netgauge/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"


# make clean

# ./configure --with-mpi=/home/liuyao/software/mpich4_1_1 \
#   LDFLAGS='-L/home/liuyao/software/Netgauge -L/home/liuyao/software/Netgauge/libnccl -lMyNcclCode -L/home/liuyao/NCCL/deps-nccl/nccl/build/lib -lnccl -L/home/liuyao/software/cuda-11.6/lib64 -lcudart -L/home/liuyao/software/mpich4_1_1/lib -lmpi -L//usr/lib64 -lpthread'

# ./configure --with-mpi=${MPI_HOME} \
#    LDFLAGS='-L${NETGAUGE_HOME} -L${libnccl_HOME} -lMyNcclCode -L${NCCL_HOME}/lib -lnccl -L${CUDA_HOME}/lib64 -lcudart -L${MPI_HOME}/lib -lmpi -L/usr/lib64 -lpthread'

export LDFLAGS="-L${NETGAUGE_HOME} -L${CUDA_HOME}/lib64 -L${MPI_HOME}/lib -L/usr/lib64"
export LIBS="-lcudart -lmpi -lpthread"
./configure --with-mpi=${MPI_HOME}

   


# sed -i 's/CXXFLAGS = -g -O2/CXXFLAGS = -g -O0/' Makefile

# sed -i 's/CFLAGS =/CFLAGS = -g -O0/' Makefile

# sed -i 's/CPPFLAGS =/CPPFLAGS = -g -O0/' Makefile 
  
# sed -i 's/CXXFLAGS = -g -O2/CXXFLAGS = -g -O0/' /home/liuyao/software/Netgauge/wnlib/Makefile 

make
