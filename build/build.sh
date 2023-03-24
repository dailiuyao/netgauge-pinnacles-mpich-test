#!/bin/bash
#SBATCH --nodes=2
#SBATCH --nodelist=gnode006,gnode008
#SBATCH --partition gpu 
##SBATCH --qos=gpu-ext
#SBATCH --time=0-00:29:00 
#SBATCH --ntasks-per-node=56 
#SBATCH --output=ng_logGP_internode%j.stdout    
#SBATCH --job-name=ng_logGP_internode   
#SBATCH --gres=gpu:2

# ---[ Script Setup ]---

set -e


module load mpich

mkdir /home/ldai8/software

cd /home/ldai8/software

git clone https://github.com/npe9/Netguage.git

./configure --prefix=/opt/apps/mpi/mpich-3.4.2_nvidiahpc-21.9-0

make