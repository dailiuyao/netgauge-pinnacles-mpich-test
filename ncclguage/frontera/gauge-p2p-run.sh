#!/bin/bash

#SBATCH -J nccl-gauge          # Job name
#SBATCH -o ./nccl-gauge.o%j       # Name of stdout output file
#SBATCH -e ./nccl-gauge.e%j       # Name of stderr error file
#SBATCH -p rtx           # Queue (partition) name
#SBATCH -N 2              # Total # of nodes (must be 1 for serial)
#SBATCH -n 4              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
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

export NCCL_DEBUG="TRACE"
export NCCL_PROTO="Simple"

cd $NCCL_GAUGE_HOME/frontera

export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/frontera"
export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"

export ITERATION_TIME="1"

export GAUGE_MIN_NTHREADS=256
export GAUGE_MAX_NTHREADS=256

export GAUGE_MIN_NCHANNELS=2
export GAUGE_MAX_NCHANNELS=2


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
                            export GAUGE_STEP_SIZE="1000"
                            NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            export GAUGE_STEP_SIZE="4"
                            for ((msize=${GAUGE_STEP_SIZE}; msize<128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                                # ibrun -n 2 --ntasks-per-node=2 \
                                # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                                # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            done
                            export GAUGE_STEP_SIZE="512"
                            for ((msize=${GAUGE_STEP_SIZE}; msize<=128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                NCCL_PROTO="Simple" ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                                # ibrun -n 2 --ntasks-per-node=2 \
                                # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                                # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            done
                        done
                    done
                done
            done 
        done
    done



#     # NCCL source location
#     NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile_D"

#     # Update to include the correct path for NVCC and MPI library paths
#     export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
#     export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

#     # for sync_mode in sync group; do
#     for sync_mode in sync; do
#         for ((n = 1; n <= 1; n *= 8)); do
#             for ((nch = ${GAUGE_MIN_NCHANNELS}; nch <= ${GAUGE_MAX_NCHANNELS}; nch *= 2)); do
#                 for mode in pping; do
#                     for ((nth = ${GAUGE_MIN_NTHREADS}; nth <= ${GAUGE_MAX_NTHREADS}; nth *= 2)); do
#                         for ((d = 2000; d <= 2000; d += 2000)); do
#                             export GAUGE_ITERATION=${itr} 
#                             export GAUGE_NCHANNELS=${nch}
#                             export GAUGE_MODE=${mode}
#                             export NCCL_MIN_NCHANNELS=${nch}
#                             export NCCL_MAX_NCHANNELS=${nch}
#                             export GAUGE_MESSAGE_SIZE=1
#                             export NCCL_NTHREADS=${nth}
#                             export GAUGE_STEP_SIZE="1000"
#                             ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
#                             # export GAUGE_STEP_SIZE="4"
#                             # for ((msize=${GAUGE_STEP_SIZE}; msize<128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
#                             #     export GAUGE_MESSAGE_SIZE=${msize}
#                             #     ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
#                             #     # ibrun -n 2 --ntasks-per-node=2 \
#                             #     # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
#                             #     # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
#                             # done
#                             # export GAUGE_STEP_SIZE="512"
#                             # for ((msize=${GAUGE_STEP_SIZE}; msize<=128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
#                             #     export GAUGE_MESSAGE_SIZE=${msize}
#                             #     ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
#                             #     # ibrun -n 2 --ntasks-per-node=2 \
#                             #     # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
#                             #     # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
#                             # done
#                         done
#                     done
#                 done
#             done 
#         done
#     done

done

# kill %1