#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition long
#SBATCH --time=0-00:29:00 
#SBATCH --ntasks-per-node=16 
#SBATCH --output=build_netgauge%j.stdout    
#SBATCH --job-name=build_netgauge   

# ---[ Script Setup ]---

set -e


module load mpich

cd /home/ldai8/software

git clone https://github.com/dailiuyao/Netgauge.git

./configure --prefix=/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0

make