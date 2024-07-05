#!/bin/bash

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

echo "mpirun -np 2 --map-by ppr:2:node ${NETGAUGE_HOME}/netgauge -m mpi -x loggp -o ng_logGP_intranode"

# dool --time --mem --cpu --net -N ib0,ens786f1,lo,total 1 > /home/liuyao/sbatch_sh/netgauge/run/output/CPU.csv  &
#         nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > /home/liuyao/sbatch_sh/netgauge/run/output/GPU.csv &
#         sh rtop.sh -d ib0 > /home/liuyao/sbatch_sh/netgauge/run/output/RTOP.csv  &

mpirun -np 2 --map-by ppr:2:node ${NETGAUGE_HOME}/netgauge -m mpi -x loggp -o ng_logGP_intranode -s 1-134217728 > /u/ldai1/ccl-build/netgauge-test/delta/run/output/ng_logGP_intranode_mpi.log



