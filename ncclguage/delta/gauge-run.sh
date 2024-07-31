#!/bin/bash
#SBATCH --job-name="a.out_symmetric"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=32  # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcjd-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 00:59:59


set -e

module load cuda
module load openmpi

export MPI_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openmpi-4.1.6-lranp74
export CUDA_HOME=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc

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

export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
export NCCL_GAUGE_HOME="/u/ldai1/ccl-build/netgauge-test/ncclguage"

export NCCL_DEBUG="TRACE"
export NCCL_PROTO="Simple"

cd $NCCL_GAUGE_HOME/delta

export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/delta"
export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"

export ITERATION_TIME="1"

export GAUGE_MIN_NTHREADS=256
export GAUGE_MAX_NTHREADS=256

export GAUGE_MIN_NCHANNELS=1
export GAUGE_MAX_NCHANNELS=1

export NCCL_SOCKET_IFNAME=hsn0.561

export COMM_GPU_ID="3"

# benchmarks for G g o L

# /home1/09168/ldai1/bin/dool --time --mem --cpu --net -N eno1,ib0,lo,total 1 > $NCCL_GAUGE_HOME/frontera/dool.csv &

for ((itr = 0; itr < ${ITERATION_TIME}; itr += 1)); do

    # # NCCL source location
    # NCCL_SRC_LOCATION="/u/ldai1/ccl-build/NCCL_profile"

    # # Update to include the correct path for NVCC and MPI library paths
    # export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
    # export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

    # # for sync_mode in sync group; do
    # for sync_mode in sync; do
    #     for ((n = 1; n <= 1; n *= 8)); do
    #         for ((nch = ${GAUGE_MIN_NCHANNELS}; nch <= ${GAUGE_MAX_NCHANNELS}; nch *= 2)); do
    #             for mode in pping; do
    #                 for ((nth = ${GAUGE_MIN_NTHREADS}; nth <= ${GAUGE_MAX_NTHREADS}; nth *= 2)); do
    #                     for ((d = 0; d <= 0; d += 1)); do
    #                         export GAUGE_ITERATION=${itr} 
    #                         export GAUGE_NCHANNELS=${nch}
    #                         export GAUGE_MODE=${mode}
    #                         export NCCL_MIN_NCHANNELS=${nch}
    #                         export NCCL_MAX_NCHANNELS=${nch}
    #                         export GAUGE_MESSAGE_SIZE=1
    #                         export NCCL_NTHREADS=${nth}
    #                         export GAUGE_STEP_SIZE="0"
    #                         $MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
    #                         export GAUGE_STEP_SIZE="32"
    #                         for ((msize=(${GAUGE_STEP_SIZE}/8); msize<4*${GAUGE_STEP_SIZE}; msize+=(${GAUGE_STEP_SIZE}/8))); do
    #                             export GAUGE_MESSAGE_SIZE=${msize}
    #                             $MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
    #                             # ibrun -n 2 --ntasks-per-node=2 \
    #                             # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
    #                             # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
    #                         done
    #                         export GAUGE_STEP_SIZE="64"
    #                         for ((msize=(${GAUGE_STEP_SIZE}*2); msize<16*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
    #                             export GAUGE_MESSAGE_SIZE=${msize}
    #                             $MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
    #                             # ibrun -n 2 --ntasks-per-node=2 \
    #                             # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
    #                             # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
    #                         done
    #                         export GAUGE_STEP_SIZE="128"
    #                         for ((msize=(${GAUGE_STEP_SIZE}*8); msize<=128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
    #                             export GAUGE_MESSAGE_SIZE=${msize}
    #                             $MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
    #                             # ibrun -n 2 --ntasks-per-node=2 \
    #                             # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
    #                             # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
    #                         done
    #                         export GAUGE_STEP_SIZE="1048576"
    #                         for ((msize=(${GAUGE_STEP_SIZE}); msize<=16*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
    #                             export GAUGE_MESSAGE_SIZE=${msize}
    #                             $MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
    #                             # ibrun -n 2 --ntasks-per-node=2 \
    #                             # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
    #                             # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
    #                         done
    #                     done
    #                 done
    #             done
    #         done 
    #     done
    # done



    # NCCL source location
    NCCL_SRC_LOCATION="/u/ldai1/ccl-build/NCCL_profile_D"

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
                            export GAUGE_STEP_SIZE="1000"
                            $MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            export GAUGE_STEP_SIZE="32"
                            for ((msize=(${GAUGE_STEP_SIZE}/8); msize<4*${GAUGE_STEP_SIZE}; msize+=(${GAUGE_STEP_SIZE}/8))); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                $MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                                # ibrun -n 2 --ntasks-per-node=2 \
                                # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                                # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            done
                            export GAUGE_STEP_SIZE="64"
                            for ((msize=(${GAUGE_STEP_SIZE}*2); msize<16*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                $MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                                # ibrun -n 2 --ntasks-per-node=2 \
                                # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                                # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            done
                            export GAUGE_STEP_SIZE="128"
                            for ((msize=(${GAUGE_STEP_SIZE}*8); msize<=128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                $MPI_HOME/bin/mpirun -n 2 --map-by ppr:1:node $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
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

done