#!/bin/bash
#SBATCH -N 1 # request 1 nodes
##SBATCH --nodelist=node01,node02
#SBATCH --output=netgauge_build_%j.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "netgauge_build"    # this is your job’s name
#SBATCH --gpus-per-node=1

# ---[ Script Setup ]---

set -e



# source ~/sbatch_sh/.nvccrc

module load nvidiahpc/21.9-0


export LD_LIBRARY_PATH="/home/liuyao/software/Netgauge/libnccl:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH=/home/liuyao/software/Netgauge/libnccl:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/liuyao/software/Netgauge/libnccl:$CPLUS_INCLUDE_PATH

export NCCL_HOME=/home/liuyao/NCCL/deps-nccl/nccl/build
# export NCCL_TEST_HOME=/home/ldai8/NCCL/deps-nccl/nccl-tests/build
export C_INCLUDE_PATH="${NCCL_HOME}/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${NCCL_HOME}/include:$CPLUS_INCLUDE_PATH"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:$LD_LIBRARY_PATH"

export CUDA_HOME=/home/liuyao/software/cuda-11.6
export PATH=/home/liuyao/software/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/home/liuyao/software/cuda-11.6/lib64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/home/liuyao/software/cuda-11.6/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/liuyao/software/cuda-11.6/include:$CPLUS_INCLUDE_PATH

export LD_LIBRARY_PATH=/home/liuyao/software/mpich_4_1_1_pgcc/lib:$LD_LIBRARY_PATH
export PATH=/home/liuyao/software/mpich_4_1_1_pgcc/bin:$PATH
export C_INCLUDE_PATH=/home/liuyao/software/mpich_4_1_1_pgcc/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/liuyao/software/mpich_4_1_1_pgcc/include:$CPLUS_INCLUDE_PATH

MPI_HOME="/home/liuyao/software/mpich_4_1_1_pgcc"
export MPI_HOME

export PATH=/home/liuyao/software/autoconf/bin:$PATH


echo "########## ENVIRONMENT ########"
echo "NCCL_LOCATION=${NCCL_HOME}"


cd /home/liuyao/software/Netgauge

autoconf --version

# export CFLAGS=" -I/home/liuyao/software/Netgauge/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"
# export CPPFLAGS=" -I/home/liuyao/software/Netgauge/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"
# export CXXFLAGS=" -I/home/liuyao/software/Netgauge/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"


make clean

# ./configure --with-mpi=/home/liuyao/software/mpich4_1_1 \
#   LDFLAGS='-L/home/liuyao/software/Netgauge -L/home/liuyao/software/Netgauge/libnccl -lMyNcclCode -L/home/liuyao/NCCL/deps-nccl/nccl/build/lib -lnccl -L/home/liuyao/software/cuda-11.6/lib64 -lcudart -L/home/liuyao/software/mpich4_1_1/lib -lmpi -L//usr/lib64 -lpthread'

./configure --with-mpi=/home/liuyao/software/mpich_4_1_1_pgcc \
   LDFLAGS='-L/home/liuyao/software/Netgauge -L/home/liuyao/software/Netgauge/libnccl -lMyNcclCode -L/home/liuyao/NCCL/deps-nccl/nccl/build/lib -lnccl -L/home/liuyao/software/cuda-11.6/lib64 -lcudart -L/home/liuyao/software/mpich_4_1_1_pgcc/lib -lmpi -L//usr/lib64 -lpthread'


   


# sed -i 's/CXXFLAGS = -g -O2/CXXFLAGS = -g -O0/' Makefile

# sed -i 's/CFLAGS =/CFLAGS = -g -O0/' Makefile

# sed -i 's/CPPFLAGS =/CPPFLAGS = -g -O0/' Makefile 
  
# sed -i 's/CXXFLAGS = -g -O2/CXXFLAGS = -g -O0/' /home/liuyao/software/Netgauge/wnlib/Makefile 

make