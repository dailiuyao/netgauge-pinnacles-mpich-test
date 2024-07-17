#!/bin/bash

#SBATCH -J nccl-gauge          # Job name
#SBATCH -o ./nccl-gauge.o%j       # Name of stdout output file
#SBATCH -e ./nccl-gauge.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 3              # Total # of nodes (must be 1 for serial)
#SBATCH -n 12              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
##SBATCH --exclude=c197-072,c196-102
##SBATCH --mail-type=all    # Send email at begin and end of job
##SBATCH -A nccl-gauge       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

set -e

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

# NCCL source location
NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile"

export NCCL_GAUGE_HOME="/home1/09168/ldai1/ccl-build/msccl_tools_lyd/examples/scripts/ncclguage"

# Update to include the correct path for NVCC and MPI library paths
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=1

# export NCCL_NTHREADS=256

export NCCL_DEBUG=INFO

cd $NCCL_GAUGE_HOME/frontera

export GAUGE_OUT_DIRE=$NCCL_GAUGE_HOME/frontera
export GAUGE_HEO="intra"
export GAUGE_CHUNK_SIZE="2"

for ((itr = 0; itr < 1; itr += 1)); do
    for ((nch = 1; nch <= 1; nch *= 2)); do
        for mode in allreduce; do
            for ((msize=1; msize<=1; msize*=2)); do
                export GAUGE_MESSAGE_SIZE=${msize}
                export GAUGE_ITERATION=${itr} 
                export GAUGE_NCHANNELS=${nch}
                export GAUGE_MODE=${mode}
                export NCCL_MIN_NCHANNELS=${nch}
                export NCCL_MAX_NCHANNELS=${nch}
                # ibrun -n 2 --ntasks-per-node=2 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                ibrun -n 6 --ntasks-per-node=2 \
                bash -c "nsys profile --force-overwrite true -o profile_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge.exe"
            done
        done
    done 
done









# ##################################### NCCL #####################################
# echo "##################################### NCCL #####################################"

# NCCLTESTS_SRC_LOCATION="/home/liuyao/scratch/deps/nccl-tests"
# export NCCLTESTS_SRC_LOCATION

# export LD_LIBRARY_PATH="${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"

# export NCCL_DEBUG=TRACE
# export NCCL_ALGO=RING
# export NCCL_PROTO=Simple
# # export NCCL_NTHREADS=192

# export NCCL_MIN_NCHANNELS=1
# export NCCL_MAX_NCHANNELS=1

# $MPI_HOME/bin/mpirun -np 2 -hosts node05:2 $NCCLTESTS_SRC_LOCATION/build/sendrecv_perf -b 4MB -e 4MB -f 2 -g 1 -n 20 > output.log 2>&1

