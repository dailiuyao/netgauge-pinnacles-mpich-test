#!/bin/bash

#SBATCH -J nccl-gauge          # Job name
#SBATCH -o ./nccl-gauge.o%j       # Name of stdout output file
#SBATCH -e ./nccl-gauge.e%j       # Name of stderr error file
#SBATCH -p rtx-dev           # Queue (partition) name
#SBATCH -N 2              # Total # of nodes (must be 1 for serial)
#SBATCH --ntasks-per-node 1  
#SBATCH -t 01:59:59        # Run time (hh:mm:ss)
#SBATCH --exclusive
##SBATCH -A nccl-gauge       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --exclude=c197-072,c196-102
##SBATCH --mail-type=all    # Send email at begin and end of job
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

export NCCL_GAUGE_HOME="/home1/09168/ldai1/ccl-build/netgauge-test/ncclguage"

export NCCL_DEBUG="DEBUG"
export NCCL_PROTO="Simple"

cd $NCCL_GAUGE_HOME/frontera

export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/frontera"
export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"

export ITERATION_TIME="100"

export COMM_GPU_ID="0"

export GAUGE_MIN_NTHREADS=64
export GAUGE_MAX_NTHREADS=64

export GAUGE_MIN_NCHANNELS=1
export GAUGE_MAX_NCHANNELS=1

GAUGE_STEP_SIZE_SMALL=32
GAUGE_STEP_SIZE_MEDIUM=64
GAUGE_STEP_SIZE_LARGE=128

if [ "$GAUGE_MAX_NCHANNELS" -eq 2 ]; then

    MESSAGE_SIZE_SMALL_START=$((GAUGE_STEP_SIZE_SMALL))
    MESSAGE_SIZE_SMALL_END=$((GAUGE_STEP_SIZE_SMALL * 8))
    MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL / 16))

    MESSAGE_SIZE_MEDIUM_START=$((GAUGE_STEP_SIZE_MEDIUM * 4))
    MESSAGE_SIZE_MEDIUM_END=$((GAUGE_STEP_SIZE_MEDIUM * 32))
    MESSAGE_SIZE_MEDIUM_STEP=$((GAUGE_STEP_SIZE_MEDIUM / 4))

    MESSAGE_SIZE_LARGE_START=$((GAUGE_STEP_SIZE_LARGE * 16))
    MESSAGE_SIZE_LARGE_END=$((GAUGE_STEP_SIZE_LARGE * 128))
    MESSAGE_SIZE_LARGE_STEP=$((GAUGE_STEP_SIZE_LARGE * 8))

elif [ "$GAUGE_MAX_NCHANNELS" -eq 1 ]; then

    MESSAGE_SIZE_SMALL_START=$((GAUGE_STEP_SIZE_SMALL))
    MESSAGE_SIZE_SMALL_END=$((GAUGE_STEP_SIZE_SMALL * 4))
    MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL / 16))

    MESSAGE_SIZE_MEDIUM_START=$((GAUGE_STEP_SIZE_MEDIUM * 2))
    MESSAGE_SIZE_MEDIUM_END=$((GAUGE_STEP_SIZE_MEDIUM * 16))
    MESSAGE_SIZE_MEDIUM_STEP=$((GAUGE_STEP_SIZE_MEDIUM / 4))

    MESSAGE_SIZE_LARGE_START=$((GAUGE_STEP_SIZE_LARGE * 8))
    MESSAGE_SIZE_LARGE_END=$((GAUGE_STEP_SIZE_LARGE * 128))
    MESSAGE_SIZE_LARGE_STEP=$((GAUGE_STEP_SIZE_LARGE * 8))

fi


# benchmarks for G g o L

# /home1/09168/ldai1/bin/dool --time --mem --cpu --net -N eno1,ib0,lo,total 1 > $NCCL_GAUGE_HOME/frontera/dool.csv &

for ((itr = 0; itr < ${ITERATION_TIME}; itr += 1)); do

    # NCCL source location
    NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile"

    # Update to include the correct path for NVCC and MPI library paths
    export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

    # for sync_mode in sync group; do
    for sync_mode in sync; do
        for ((n = 1; n <= 1; n *= 8)); do
            for ((nch = ${GAUGE_MIN_NCHANNELS}; nch <= ${GAUGE_MAX_NCHANNELS}; nch *= 2)); do
                for mode in pping; do
                    for ((nth = ${GAUGE_MIN_NTHREADS}; nth <= ${GAUGE_MAX_NTHREADS}; nth *= 2)); do
                        for ((d = 0; d <= 0; d += 1)); do
                            export GAUGE_ITERATION=${itr} 
                            export GAUGE_NCHANNELS=${nch}
                            export GAUGE_MODE=${mode}
                            export NCCL_MIN_NCHANNELS=${nch}
                            export NCCL_MAX_NCHANNELS=${nch}
                            export GAUGE_MESSAGE_SIZE=1
                            export NCCL_NTHREADS=${nth}

                            # export GAUGE_STEP_SIZE="0"
                            # NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            # export GAUGE_STEP_SIZE="32"
                            # for ((msize=${MESSAGE_SIZE_SMALL_START}; msize<${MESSAGE_SIZE_SMALL_END}; msize+=${MESSAGE_SIZE_SMALL_STEP})); do
                            #     export GAUGE_MESSAGE_SIZE=${msize}
                            #     NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            # done
                            export GAUGE_STEP_SIZE="64"
                            for ((msize=${MESSAGE_SIZE_MEDIUM_START}; msize<${MESSAGE_SIZE_MEDIUM_END}; msize+=${MESSAGE_SIZE_MEDIUM_STEP})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            done 
                            export GAUGE_STEP_SIZE="128"
                            for ((msize=${MESSAGE_SIZE_LARGE_START}; msize<=${MESSAGE_SIZE_LARGE_END}; msize+=${MESSAGE_SIZE_LARGE_STEP})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            done
                        done
                    done
                done
            done 
        done
    done



    # NCCL source location
    NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile_D"

    # Update to include the correct path for NVCC and MPI library paths
    export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

    # for sync_mode in sync group; do
    for sync_mode in sync; do
        for ((n = 1; n <= 1; n *= 8)); do
            for ((nch = ${GAUGE_MIN_NCHANNELS}; nch <= ${GAUGE_MAX_NCHANNELS}; nch *= 2)); do
                for mode in pping; do
                    for ((nth = ${GAUGE_MIN_NTHREADS}; nth <= ${GAUGE_MAX_NTHREADS}; nth *= 2)); do
                        for ((d = 2000; d <= 2000; d += 2000)); do
                            export GAUGE_ITERATION=${itr} 
                            export GAUGE_NCHANNELS=${nch}
                            export GAUGE_MODE=${mode}
                            export NCCL_MIN_NCHANNELS=${nch}
                            export NCCL_MAX_NCHANNELS=${nch}
                            export GAUGE_MESSAGE_SIZE=1
                            export NCCL_NTHREADS=${nth}

                            # export GAUGE_STEP_SIZE="0"
                            # NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            # export GAUGE_STEP_SIZE="32"
                            # for ((msize=${MESSAGE_SIZE_SMALL_START}; msize<${MESSAGE_SIZE_SMALL_END}; msize+=${MESSAGE_SIZE_SMALL_STEP})); do
                            #     export GAUGE_MESSAGE_SIZE=${msize}
                            #     NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            # done
                            export GAUGE_STEP_SIZE="64"
                            for ((msize=${MESSAGE_SIZE_MEDIUM_START}; msize<${MESSAGE_SIZE_MEDIUM_END}; msize+=${MESSAGE_SIZE_MEDIUM_STEP})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            done 
                            export GAUGE_STEP_SIZE="128"
                            for ((msize=${MESSAGE_SIZE_LARGE_START}; msize<=${MESSAGE_SIZE_LARGE_END}; msize+=${MESSAGE_SIZE_LARGE_STEP})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            done
                        done
                    done
                done
            done 
        done
    done

done

# kill %1