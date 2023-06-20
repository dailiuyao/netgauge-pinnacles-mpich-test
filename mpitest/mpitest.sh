#!/bin/bash
#SBATCH -N 2 # request 1 nodes
#SBATCH --nodelist=node01,node02
#SBATCH --output=netgauge_test.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "netgauge_test"    # this is your job’s name
#SBATCH --gpus-per-node=1


# source /home/liuyao/sbatch_sh/.nvccrc

# source /home/liuyao/sbatch_sh/.openmpi-2.1.6

# source ~/sbatch_sh/.openmpirc

# module load openmpi/4.1.1-gcc-8.4.1

# mpirun -n 2 -host node01,node02 -np 1 /home/liuyao/software/Netgauge/netgauge-2.4.6/netgauge -h 

# mpirun -host node01 --map-by=ppr:1:node -host /home/liuyao/software/Netgauge/netgauge-2.4.6/netgauge --verbosity 3 -t 30 -s 1048576 -c 20 -m ib 

# mpirun -host node01 -np 1 /home/liuyao/software/Netgauge/netgauge-2.4.6/netgauge --verbosity 3 -t 30 -s 1048576 -c 20 -m ib 

# mpirun -n 1 /home/liuyao/software/Netgauge/netgauge-2.4.6/netgauge --verbosity 3 -t 30 -s 1048576 -c 20 -m ib 

# mpirun \
#  --mca openib,self \
#  --map-by ppr:1:node \
#  --host "node01" \
#  /home/liuyao/software/Netgauge/netgauge-2.4.6/netgauge --verbosity 3 --servermode -t 30 -s 1048576 -c 20 -m mpi : \
#  --mca openib,self \
#  --map-by ppr:1:node \
#  --host "node02" \
#  /home/liuyao/software/Netgauge/netgauge-2.4.6/netgauge --verbosity 3 -t 30 -s 1048576 -c 20 -m mpi


#  mpirun \
#  --map-by ppr:1:node \
#  --host "node01" \
#  /home/liuyao/software/Netgauge/netgauge-2.4.6/netgauge --verbosity 100 --hostnames --sanity-check -t 30 -s 1048576 -c 20 -m ib


module load mpich/3.4.2-nvidiahpc-21.9-0

mpirun \
 --verbose \
 -np 1 \
 --host "node01" \
 /home/liuyao/software/Netgauge/netgauge-2.4.6/netgauge --verbosity 3 --servermode -t 30 -s 1048576 -c 20 -m mpi : \
 --verbose \
 -np 1 \
 --host "node02" \
 /home/liuyao/software/Netgauge/netgauge-2.4.6/netgauge --verbosity 3 -t 30 -s 1048576 -c 20 -m mpi



# mpirun --verbose --map-by ppr:1:node --host "node01,node02" mpi

# mpirun --map-by ppr:1:node --host "node01,node02" mpi