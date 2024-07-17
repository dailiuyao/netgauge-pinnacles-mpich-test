#!/bin/bash

#SBATCH -J ccl-run           # Job name
#SBATCH -o ./log/ccl-run-common.o%j       # Name of stdout output file
#SBATCH -e ./log/ccl-run-common.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 2             # Total # of nodes (must be 1 for serial)
#SBATCH -n 2               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 23:59:59        # Run time (hh:mm:ss)
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A ccl-run-common       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu


# ---[ Script Setup ]---

set -e

# module load mpich

module load impi/19.0.5
module load cuda/11.3
module load intel  

export CUDA_HOME=/opt/apps/cuda/11.3
# export MPI_HOME=/scratch1/projects/compilers/intel18u5/compilers_and_libraries_2018.6.288/linux/mpi/intel64
export MPI_HOME=/opt/intel/compilers_and_libraries_2019.5.281/linux/mpi/intel64

export NVCC_GENCODE="-gencode=arch=compute_75,code=sm_75"

# Update to include the correct path for MPI library paths
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export PATH=$CUDA_HOME/bin:$PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDNN_LIBRARY=$CUDA_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDA_HOME/include


export NETGAUGE_HOME="/home1/09168/ldai1/ccl-build/netgauge_mpi"

echo "ibrun -n 2 --ntasks-per-node=1 ${NETGAUGE_HOME}/netgauge -m mpi -x loggp -o ng_logGP_internode"

export OUTPUT_DIR="/home1/09168/ldai1/ccl-build/netgauge-test/frontera/run/output"

# dool --time --mem --cpu --net -N hsn0,lo,total 1 > ${OUTPUT_DIR}/CPU.csv  &
#         nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > ${OUTPUT_DIR}/GPU.csv &
#         sh rtop.sh -d hsn0 > ${OUTPUT_DIR}/RTOP.csv  &

# export UCX_NET_DEVICES=ib0


ibrun -n 2 --ntasks-per-node=1 ${NETGAUGE_HOME}/netgauge -m mpi -x loggp -o ng_logGP_intranode --size=1048576-1073741824 > ${OUTPUT_DIR}/ng_logGP_internode_mpi.log



