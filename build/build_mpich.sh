#!/bin/bash
#SBATCH -N 1 # request 1 nodes
##SBATCH --nodelist=node01,node02
#SBATCH --output=netgauge_build_%j.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "netgauge_build"    # this is your job’s name
#SBATCH --gpus-per-node=1

# ---[ Script Setup ]---
set -e

module load nvidiahpc/21.9-0

pgcc --version

source /home/liuyao/sbatch_sh/.nvccrc

cd /home/liuyao/mpich-4.1.1

make clean

./configure CC=pgcc --prefix=/home/liuyao/software/mpich_4_1_1_pgcc --with-cuda=/home/liuyao/software/cuda-11.6

make -j16

make install