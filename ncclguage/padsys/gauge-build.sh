#!/bin/bash

# Set environment variables

spack load gcc@10.4.0 

# spack load mpich@4.1.1 

# export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-10.4.0/mpich-4.1.1-j7lgvgtzrx6aj5k6a7lcs5xg4obnfi6i"

spack load openmpi@5.0.3

export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-8.5.0/openmpi-5.0.3-xsxjs6lg2gnrmhfygn5bpoyaeadarmcl"

export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

source /home/liuyao/sbatch_sh/.nvccrc

# NCCL source location
export NCCL_SRC_LOCATION="/home/liuyao/scratch/deps/NCCL_profile"

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

export NCCL_GAUGE_HOME="/home/liuyao/scratch/deps/netgauge-test/ncclguage"

concurrency_sequence=(1 2 4 8 16)

experiments_number=5

for ((e = 0; e < experiments_number; e += 1)); do
    for i in "${concurrency_sequence[@]}"; do
        for mode in pping; do
            # Use proper variable expansion and quoting in the command
            nvcc "$NVCC_GENCODE" -ccbin g++ -I"${NCCL_SRC_LOCATION}/build/include" -I"${MPI_HOME}/include" \
                -L"${NCCL_SRC_LOCATION}/build/lib" -L"${CUDA_HOME}/lib64" -L"${MPI_HOME}/lib" -lnccl -lcudart -lmpi \
                -D N_ITERS=${i} \
                "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge.cu" -o "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_e_${e}.exe"

            # Verification of the output
            if [ -f "${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_e_${e}.exe" ]; then
                echo "Compilation successful. Output file: ${NCCL_GAUGE_HOME}/gauge/${mode}_gauge_n_${i}_e_${e}.exe"
            else
                echo "Compilation failed."
            fi
        done
    done
done