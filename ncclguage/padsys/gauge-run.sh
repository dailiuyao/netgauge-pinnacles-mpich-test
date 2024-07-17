#!/usr/bin/env bash
#SBATCH --partition=A100
#SBATCH --nodes=2 # request 1 nodes
#SBATCH --nodelist=node01,node02
#SBATCH --output=ncclgauge-run.stdout    # standard output will be redirected to this file, where the % is replaced with the job allocation number.
#SBATCH -J "ncclgauge-run"    # this is your job’s name
#SBATCH --gpus-per-node=1

# Set environment variables

spack load gcc@10.4.0 

spack load openmpi@5.0.3

export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-10.4.0/openmpi-5.0.3-ltv5k5ckeuhvwzb2dnjqsb22eggfhmwh"

# spack load mpich@4.1.1 

# export MPI_HOME="/home/liuyao/software/spack/opt/spack/linux-almalinux8-icelake/gcc-10.4.0/mpich-4.1.1-j7lgvgtzrx6aj5k6a7lcs5xg4obnfi6i"

export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

source /home/liuyao/sbatch_sh/.nvccrc

# Ensure known hosts are updated
ssh-keyscan -H node01 node02 >> ~/.ssh/known_hosts

# Helper functions
mpiwrap() {
  $MPI_HOME/bin/mpirun -host node01,node02 --map-by ppr:1:node \
  --mca btl_tcp_if_include en2 \
  --mca btl tcp,self --bind-to none \
  -x MPI_HOME -x CUDA_HOME -x PATH -x LD_LIBRARY_PATH -x C_INCLUDE_PATH \
  -x NCCL_DEBUG -x NCCL_PROTO \
  -x GAUGE_OUT_DIRE -x GAUGE_HEO -x GAUGE_CHUNK_SIZE \
  -x GAUGE_ITERATION -x GAUGE_NCHANNELS -x GAUGE_MODE -x NCCL_MIN_NCHANNELS -x NCCL_MAX_NCHANNELS -x NCCL_NTHREADS -x GAUGE_STEP_SIZE \
  -x NCCL_SOCKET_IFNAME $@
}

# mpiwrap() {
#   $MPI_HOME/bin/mpiexec -hosts node01,node02 -ppn 1 -bind-to none \
#   $@
# }


mpiwrap hostname

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

NCCL_GAUGE_HOME="/home/liuyao/scratch/deps/netgauge-test/ncclguage"

export NCCL_DEBUG=TRACE
export NCCL_PROTO=Simple

cd $NCCL_GAUGE_HOME/padsys

export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/padsys"
export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"

ITERATION_TIME="2"

GAUGE_MIN_NTHREADS=256
GAUGE_MAX_NTHREADS=256

GAUGE_MIN_NCHANNELS=2
GAUGE_MAX_NCHANNELS=2

# export UCX_NET_DEVICES=ib0

export NCCL_SOCKET_IFNAME=ib0

# benchmarks for G g o L

sh $NCCL_GAUGE_HOME/rtop.sh -d ib0 > ${GAUGE_OUT_DIRE}/RTOP.csv  &

for ((itr = 0; itr < ${ITERATION_TIME}; itr += 1)); do

    # NCCL source location
    export NCCL_SRC_LOCATION="/home/liuyao/scratch/deps/NCCL_profile"

    # Update to include the correct path for NVCC and MPI library paths
    export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${LD_LIBRARY_PATH}
    export PATH="${NCCL_SRC_LOCATION}/build/include:${PATH}"

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
                            mpiwrap $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            export GAUGE_STEP_SIZE="4"
                            for ((msize=${GAUGE_STEP_SIZE}; msize<128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                mpiwrap $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                                # ibrun -n 2 --ntasks-per-node=2 \
                                # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                                # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            done
                            export GAUGE_STEP_SIZE="512"
                            for ((msize=${GAUGE_STEP_SIZE}; msize<=128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                mpiwrap $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
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



    # # NCCL source location
    # NCCL_SRC_LOCATION="/home1/09168/ldai1/ccl-build/NCCL_profile_D"

    # # Update to include the correct path for NVCC and MPI library paths
    # export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
    # export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

    # # for sync_mode in sync group; do
    # for sync_mode in sync; do
    #     for ((n = 1; n <= 1; n *= 8)); do
    #         for ((nch = ${GAUGE_MIN_NCHANNELS}; nch <= ${GAUGE_MAX_NCHANNELS}; nch *= 2)); do
    #             for mode in pping; do
    #                 for ((nth = ${GAUGE_MIN_NTHREADS}; nth <= ${GAUGE_MAX_NTHREADS}; nth *= 2)); do
    #                     for ((d = 2000; d <= 2000; d += 2000)); do
    #                         export GAUGE_ITERATION=${itr} 
    #                         export GAUGE_NCHANNELS=${nch}
    #                         export GAUGE_MODE=${mode}
    #                         export NCCL_MIN_NCHANNELS=${nch}
    #                         export NCCL_MAX_NCHANNELS=${nch}
    #                         export GAUGE_MESSAGE_SIZE=1
    #                         export NCCL_NTHREADS=${nth}
    #                         export GAUGE_STEP_SIZE="1000"
    #                         ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
    #                         export GAUGE_STEP_SIZE="4"
    #                         for ((msize=${GAUGE_STEP_SIZE}; msize<128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
    #                             export GAUGE_MESSAGE_SIZE=${msize}
    #                             ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
    #                             # ibrun -n 2 --ntasks-per-node=2 \
    #                             # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
    #                             # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
    #                         done
    #                         export GAUGE_STEP_SIZE="512"
    #                         for ((msize=${GAUGE_STEP_SIZE}; msize<=128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
    #                             export GAUGE_MESSAGE_SIZE=${msize}
    #                             ibrun -n 2 --ntasks-per-node=1 $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
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

done

kill %1

