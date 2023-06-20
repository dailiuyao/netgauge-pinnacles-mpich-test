#!/bin/bash
#SBATCH -N 2 # request 1 nodes
##SBATCH --nodelist=node01,node02
#SBATCH --output=./stdout/ens786f0/netgauge_run_%j.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "netgauge_run"    # this is your job’s name
#SBATCH --gpus-per-node=1

# ---[ Script Setup ]---

set -e

# module load mpich

# module load mpich/3.4.2-nvidiahpc-21.9-0

# MPI_HOME="/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0"
# export MPI_HOME
# export PATH="${MPI_HOME}/include:$PATH"
# export LD_LIBRARY_PATH="${MPI_HOME}/lib:$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH=/home/liuyao/software/mpich4_1_1/lib:$LD_LIBRARY_PATH
export PATH=/home/liuyao/software/mpich4_1_1/bin:$PATH
export C_INCLUDE_PATH=/home/liuyao/software/mpich4_1_1/include:$C_INCLUDE_PATH

MPI_HOME="/home/liuyao/software/mpich4_1_1"
export MPI_HOME

echo "UCX_NET_DEVICES=ens786f0 mpirun -n 2 /home/liuyao/software/Netgauge_default/netgauge -m mpi -x loggp -o ng_logGP_internode"

dool --time --mem --cpu --net -N ib0,ens786f0,lo,total 1 > /home/liuyao/sbatch_sh/netgauge_default/run/output/ens786f0/CPU.csv  &
        nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > /home/liuyao/sbatch_sh/netgauge_default/run/output/ens786f0/GPU.csv &
        sh rtop.sh -d ib0 > /home/liuyao/sbatch_sh/netgauge_default/run/output/ens786f0/RTOP.csv  &

mpirun -n 2 hostname

# mpirun -n 2 -host {node01,node02} /home/liuyao/software/Netgauge_default/netgauge -m mpi -x loggp -o ng_logGP_internode

### 'en2'(tcp), 'ens786f0'(tcp), 'ib0'(tcp), 'lo'(tcp), 'mlx5_0:1'(ib), 'mlx5_1:1'(ib), 'mlx5_2:1'(ib)

UCX_NET_DEVICES=ens786f0 mpirun -n 2  /home/liuyao/software/Netgauge_default/netgauge -m mpi -x loggp -o ng_logGP_internode

# mpirun -n 2 /home/liuyao/software/Netgauge_default/netgauge -m ib -x overlap -o ng_logGP_intrano



