#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:59:59
#PBS -q debug-scaling
#PBS -l filesystems=home
#PBS -A SR_APPFL 
#PBS -k doe
#PBS -N ncclgauge
#PBS -o log/ncclgauge.out
#PBS -e log/ncclgauge.error

# Set environment variables

module swap PrgEnv-nvhpc PrgEnv-gnu
module load nvhpc-mixed

# Install and load libxml2 using Spack
spack load libxml2

export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/nvidia/23.3
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda

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

export MPIEXEC_HOME=/opt/cray/pals/1.3.4
export NCCL_NET_PLUGIN_HOME="/home/ldai8/ccl/aws-ofi-nccl-1.7.4-aws/build"     
export NCCL_SOCKET_IFNAME=hsn0,hsn1
export NCCL_IB_HCA=cxi0,cxi1
export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:$LD_LIBRARY_PATH

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

export NCCL_GAUGE_HOME="/home/ldai8/ccl/netgauge-test/ncclguage"

export NCCL_DEBUG="WARN"
export NCCL_PROTO="Simple"

cd $NCCL_GAUGE_HOME/polaris

export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/polaris"
export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"

export ITERATION_TIME="30"

export COMM_GPU_ID="0"

export GAUGE_MIN_NTHREADS=64
export GAUGE_MAX_NTHREADS=64

export GAUGE_MIN_NCHANNELS=1
export GAUGE_MAX_NCHANNELS=1


# benchmarks for G g o L

# /home1/09168/ldai1/bin/dool --time --mem --cpu --net -N eno1,ib0,lo,total 1 > $NCCL_GAUGE_HOME/frontera/dool.csv &

for ((itr = 10; itr < ${ITERATION_TIME}; itr += 1)); do

    # NCCL source location
    NCCL_SRC_LOCATION="/home/ldai8/ccl/NCCL_profile"

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
                            export GAUGE_STEP_SIZE="0"
                            $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            export GAUGE_STEP_SIZE="32"
                            for ((msize=(${GAUGE_STEP_SIZE}/8); msize<4*${GAUGE_STEP_SIZE}; msize+=(${GAUGE_STEP_SIZE}/8))); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                                # ibrun -n 2 --ntasks-per-node=2 \
                                # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                                # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            done
                            export GAUGE_STEP_SIZE="64"
                            for ((msize=(${GAUGE_STEP_SIZE}*2); msize<16*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                                # ibrun -n 2 --ntasks-per-node=2 \
                                # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                                # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            done
                            # export GAUGE_STEP_SIZE="128"
                            # for ((msize=(${GAUGE_STEP_SIZE}*8); msize<=128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                            #     export GAUGE_MESSAGE_SIZE=${msize}
                            #     $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            #     # ibrun -n 2 --ntasks-per-node=2 \
                            #     # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                            #     # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            # done
                            # export GAUGE_STEP_SIZE="1048576"
                            # for ((msize=(${GAUGE_STEP_SIZE}); msize<=16*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                            #     export GAUGE_MESSAGE_SIZE=${msize}
                            #     $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            #     # ibrun -n 2 --ntasks-per-node=2 \
                            #     # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                            #     # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            # done
                        done
                    done
                done
            done 
        done
    done



    # NCCL source location
    NCCL_SRC_LOCATION="/home/ldai8/ccl/NCCL_profile_D"

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
                            $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            export GAUGE_STEP_SIZE="32"
                            for ((msize=(${GAUGE_STEP_SIZE}/8); msize<4*${GAUGE_STEP_SIZE}; msize+=(${GAUGE_STEP_SIZE}/8))); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                                # ibrun -n 2 --ntasks-per-node=2 \
                                # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                                # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            done
                            export GAUGE_STEP_SIZE="64"
                            for ((msize=(${GAUGE_STEP_SIZE}*2); msize<16*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                                export GAUGE_MESSAGE_SIZE=${msize}
                                $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                                # ibrun -n 2 --ntasks-per-node=2 \
                                # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                                # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            done
                            # export GAUGE_STEP_SIZE="128"
                            # for ((msize=(${GAUGE_STEP_SIZE}*8); msize<=128*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                            #     export GAUGE_MESSAGE_SIZE=${msize}
                            #     $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}.exe
                            #     # ibrun -n 2 --ntasks-per-node=2 \
                            #     # bash -c "nsys profile --force-overwrite true -o p2p_profile_d_0_n_${n}_${mode}_%q{SLURM_PROCID} --trace=cuda,nvtx,osrt --stats=true $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe"
                            #     # ibrun -n 2 --ntasks-per-node=2 ncu --mode=launch $NCCL_GAUGE_HOME/gauge/${mode}_gauge_${n}.exe
                            # done
                        done
                    done
                done
            done 
        done
    done

done

# kill %1