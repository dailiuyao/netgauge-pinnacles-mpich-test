#!/bin/bash
#SBATCH --nodes=2
#SBATCH --nodelist=gnode004,gnode008
#SBATCH --partition gpu 
##SBATCH --qos=gpu-ext
#SBATCH --time=0-00:59:00 
#SBATCH --ntasks-per-node=56 
#SBATCH --output=test%j.stdout    
#SBATCH --job-name=test   
#SBATCH --gres=gpu:2

# ---[ Script Setup ]---

sleep 10000000000




