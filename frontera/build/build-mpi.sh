#!/bin/bash


# ---[ Script Setup ]---

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


echo "########## ENVIRONMENT ########"

export NETGAUGE_HOME="/home1/09168/ldai1/ccl-build/netgauge_mpi"

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
