#!/bin/bash -l

set -e

module load cuda/11.3
module load intel  


export CUDA_HOME=/opt/apps/cuda/11.3

export NVCC_GENCODE="-gencode=arch=compute_75,code=sm_75"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}


./GetCudaFrequency.exe


# GPU Clock Rate: 1815000 kHz
# GPU Clock Rate: 1815000064 Hz