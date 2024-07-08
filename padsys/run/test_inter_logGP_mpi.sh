#!/bin/bash
#SBATCH -N 2 # request 1 nodes
#SBATCH --nodelist=node01,node02
#SBATCH --output=netgauge_run_%j.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "netgauge_run"    # this is your job’s name
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

export NETGAUGE_HOME="/home/liuyao/software/netgauge_mpi"
export NCCL_DEBUG=TRACE

# UCX_NET_DEVICES=ib0 $MPI_HOME/bin/mpirun --map-by ppr:1:node ${NETGAUGE_HOME}/netgauge -m mpi -x loggp -o ng_logGP_internode -s 1-134217728 > /home/liuyao/scratch/deps/netgauge-test/padsys/run/output/ng_logGP_internode_mpi.log

$MPI_HOME/bin/mpirun --map-by ppr:1:node ${NETGAUGE_HOME}/netgauge -m mpi -x loggp -o ng_logGP_internode -s 1-134217728





















# echo "mpirun -n 2 /home/liuyao/software/Netgauge/netgauge -m mpi -x loggp -o ng_logGP_internode"

# dool --time --mem --cpu --net -N ib0,ens786f1,lo,total 1 > /home/liuyao/sbatch_sh/netgauge/run/output/CPU.csv  &
#         nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > /home/liuyao/sbatch_sh/netgauge/run/output/GPU.csv &
#         sh rtop.sh -d ib0 > /home/liuyao/sbatch_sh/netgauge/run/output/RTOP.csv  &

# # mpirun -n 2 /home/ldai8/software/netgauge-2.4.6/netgauge --verbosity 3 -t 30 -s 1048576 -c 20 -g 65535 -x loggp -o

# UCX_NET_DEVICES=mlx5_0:1 mpirun -n 2  /home/liuyao/software/Netgauge_default/netgauge -h

# UCX_NET_DEVICES=mlx5_0:1 mpirun -n 2 -ppn 1 /home/liuyao/software/Netgauge/netgauge -m mpi -x loggp -o ng_logGP_internode -s 1-131072

# # mpiexec -n 2 -ppn 1 gdb --args /home/ldai8/software/Netgauge/netgauge -m mpi -x loggp -o ng_logGP_internode : -n 2 -ppn 1 \
# #  /home/ldai8/software/Netgauge/netgauge -m mpi -x loggp -o ng_logGP_internode



