#!/bin/bash
#SBATCH --job-name="a.out_symmetric"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcjd-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 04:00:00

# ---[ Script Setup ]---

set -e

# module load mpich

module load cuda
module load openmpi

export MPI_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openmpi-4.1.6-lranp74
export CUDA_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc

export LD_LIBRARY_PATH="${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
export PATH="${MPI_HOME}/bin:${CUDA_HOME}/bin:$PATH"
export C_INCLUDE_PATH="${MPI_HOME}/include:${CUDA_HOME}/include:$C_INCLUDE_PATH"


export NETGAUGE_HOME="/u/ldai1/ccl-build/netgauge_default"

echo "mpirun -np 2 --map-by ppr:1:node ${NETGAUGE_HOME}/netgauge -m mpi -x loggp -o ng_logGP_internode"

export OUTPUT_DIR="/u/ldai1/ccl-build/netgauge-test/delta/run/output"

# dool --time --mem --cpu --net -N hsn0,lo,total 1 > ${OUTPUT_DIR}/CPU.csv  &
#         nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > ${OUTPUT_DIR}/GPU.csv &
#         sh rtop.sh -d hsn0 > ${OUTPUT_DIR}/RTOP.csv  &

export NCCL_DEBUG=TRACE

UCX_NET_DEVICES=hsn0 mpirun -np 2 --map-by ppr:1:node ${NETGAUGE_HOME}/netgauge -m mpi -x loggp -o ng_logGP_intranode -s 1-134217728 > /u/ldai1/ccl-build/netgauge-test/delta/run/output/ng_logGP_internode_mpi.log



