#!/bin/bash
#SBATCH -N 1 # request 1 nodes
##SBATCH --nodelist=node01,node02
#SBATCH --output=netgauge_build_%j.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "netgauge_build"    # this is your job’s name
#SBATCH --gpus-per-node=1

# ---[ Script Setup ]---




set -e

export LD_LIBRARY_PATH=/home/liuyao/software/mpich4_1_1/lib:$LD_LIBRARY_PATH
export PATH=/home/liuyao/software/mpich4_1_1/bin:$PATH
export C_INCLUDE_PATH=/home/liuyao/software/mpich4_1_1/include:$C_INCLUDE_PATH

MPI_HOME="/home/liuyao/software/mpich4_1_1"
export MPI_HOME


autoconf --version

# source ~/sbatch_sh/.nvccrc

# CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda"
# export CUDA_HOME
# export PATH="${CUDA_HOME}/include:$PATH"
# export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

cd /home/liuyao/software/Netgauge_default

make clean

./configure --with-mpi=/home/liuyao/software/mpich4_1_1 \
 LDFLAGS='-L//usr/lib64' \
 LIBS='-lpthread'


#  CXXFLAGS='-L//usr/lib64' \

make