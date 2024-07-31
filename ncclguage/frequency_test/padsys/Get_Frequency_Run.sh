#!/bin/bash -l

set -e

source /home/liuyao/sbatch_sh/.nvccrc

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

./GetCudaFrequency.exe


