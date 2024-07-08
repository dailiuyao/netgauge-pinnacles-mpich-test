#!/bin/bash
#SBATCH -N 2 # request 1 nodes
#SBATCH --nodelist=node01,node02
#SBATCH --output=./log/netgauge_run_%j.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "netgauge_run"    # this is your job’s name
#SBATCH --exclusive
#SBATCH --gpus-per-node=1

# ---[ Script Setup ]---

set -e

spack load gcc@10.4.0 

spack load mpich@4.1.1

export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-10.4.0/mpich-4.1.1-j7lgvgtzrx6aj5k6a7lcs5xg4obnfi6i"

source /home/liuyao/sbatch_sh/.nvccrc

export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export NCCL_HOME="/home/liuyao/scratch/deps/nccl/build"
export C_INCLUDE_PATH="${NCCL_HOME}/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${NCCL_HOME}/include:$CPLUS_INCLUDE_PATH"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:$LD_LIBRARY_PATH"

export libnccl_HOME="/home/liuyao/software/netgauge_nccl/libnccl"

export LD_LIBRARY_PATH="${libnccl_HOME}:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="${libnccl_HOME}:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${libnccl_HOME}:$CPLUS_INCLUDE_PATH"

export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=1

export NCCL_DEBUG=TRACE

export NCCL_PROTO=Simple

export NETGAUGE_HOME="/home/liuyao/software/netgauge_nccl"

export OUTPUT_DIR="/home/liuyao/scratch/deps/netgauge-test/padsys/run/output/nccl"

export NETGAUGE_TEST_HOME="/home/liuyao/scratch/deps/netgauge-test/padsys"

dool --time --mem --cpu --net -N ib0,ens786f1,lo,total 1 > ${OUTPUT_DIR}/CPU.csv  &
        nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > ${OUTPUT_DIR}/GPU.csv &
        sh rtop.sh -d ib0 > ${OUTPUT_DIR}/RTOP.csv  &

UCX_NET_DEVICES=ib0 $MPI_HOME/bin/mpirun -ppn 1 ${NETGAUGE_HOME}/netgauge -m mpi -x loggp -o ng_logGP_internode --size=1048576-1073741824 > ${OUTPUT_DIR}/ng_logGP_internode_nccl.log

kill %1
kill %2
kill %3















