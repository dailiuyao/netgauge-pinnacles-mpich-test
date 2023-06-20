#!/bin/bash
#SBATCH -N 1 # request 1 nodes
##SBATCH --nodelist=node01,node02
#SBATCH --output=netgauge_build_%j.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "netgauge_build"    # this is your job’s name
#SBATCH --gpus-per-node=1

# ---[ Script Setup ]---

set -e



# source ~/sbatch_sh/.nvccrc


export LD_LIBRARY_PATH="/home/liuyao/software/Netgauge_gdr/libcuda:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH=/home/liuyao/software/Netgauge_gdr/libcuda:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/liuyao/software/Netgauge_gdr/libcuda:$CPLUS_INCLUDE_PATH

export CUDA_HOME=/home/liuyao/software/cuda-11.6
export PATH=/home/liuyao/software/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/home/liuyao/software/cuda-11.6/lib64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/home/liuyao/software/cuda-11.6/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/liuyao/software/cuda-11.6/include:$CPLUS_INCLUDE_PATH

export LD_LIBRARY_PATH=/home/liuyao/software/mpich_4_1_1_gdr/lib:$LD_LIBRARY_PATH
export PATH=/home/liuyao/software/mpich_4_1_1_gdr/bin:$PATH
export C_INCLUDE_PATH=/home/liuyao/software/mpich_4_1_1_gdr/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/home/liuyao/software/mpich_4_1_1_gdr/include:$CPLUS_INCLUDE_PATH

MPI_HOME="/home/liuyao/software/mpich_4_1_1_gdr"
export MPI_HOME

export PATH=/home/liuyao/software/autoconf/bin:$PATH


cd /home/liuyao/software/Netgauge_gdr

autoconf --version

# export CFLAGS=" -I/home/liuyao/software/Netgauge/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"
# export CPPFLAGS=" -I/home/liuyao/software/Netgauge/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"
# export CXXFLAGS=" -I/home/liuyao/software/Netgauge/libnccl -I/home/liuyao/NCCL/deps-nccl/nccl/build/include -I/home/liuyao/software/cuda-11.6/include -I/home/liuyao/software/mpich4_1_1/include"


# make clean

./configure --with-mpi=/home/liuyao/software/mpich_4_1_1_gdr \
  LDFLAGS='-L/home/liuyao/software/Netgauge_gdr -L/home/liuyao/software/Netgauge_gdr/libcuda -lMyCudaCode -L/home/liuyao/software/cuda-11.6/lib64 -lcudart -L/home/liuyao/software/mpich4_1_1_gdr/lib -lmpi -L//usr/lib64 -lpthread'

# sed -i 's/CXXFLAGS = -g -O2/CXXFLAGS = -g/' Makefile

# sed -i 's/CFLAGS =/CFLAGS = -g/' Makefile

# sed -i 's/CPPFLAGS =/CPPFLAGS = -g/' Makefile 
  
# sed -i 's/CXXFLAGS = -g -O2/CXXFLAGS = -g/' /home/liuyao/software/Netgauge/wnlib/Makefile 

make
