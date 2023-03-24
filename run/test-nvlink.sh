#!/bin/bash
#SBATCH --nodes=1
##SBATCH --nodelist=gnode006,gnode008
#SBATCH --partition gpu 
##SBATCH --qos=gpu-ext
#SBATCH --time=0-00:29:00 
#SBATCH --ntasks-per-node=56 
#SBATCH --output=test-nvlink%j.stdout    
#SBATCH --job-name=test-nvlink   
#SBATCH --gres=gpu:2

# ---[ Script Setup ]---

set -e


source /home/ldai8/bash/.nvccrc

nvidia-smi topo -m
