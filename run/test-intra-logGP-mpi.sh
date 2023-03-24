#!/bin/bash
#SBATCH --nodes=1
##SBATCH --nodelist=gnode006,gnode008
#SBATCH --partition gpu 
##SBATCH --qos=gpu-ext
#SBATCH --time=0-00:29:00 
#SBATCH --ntasks-per-node=56 
#SBATCH --output=ng_logGP_intranode%j.stdout    
#SBATCH --job-name=ng_logGP_intranode   
#SBATCH --gres=gpu:2

# ---[ Script Setup ]---

set -e


module load mpich

mpirun -n 2 hostname

# mpirun -n 2 /home/ldai8/software/Netguage/netgauge --verbosity 3 -t 30 -s 1048576 -c 20 -g 65535 -x loggp -o

mpirun -n 2 /home/ldai8/software/Netguage/netgauge -m mpi -x loggp -o ng_logGP_intranode



