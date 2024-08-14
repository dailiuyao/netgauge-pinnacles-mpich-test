#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:59:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A SR_APPFL 
#PBS -k doe
#PBS -N nething
#PBS -o log/nething.out
#PBS -e log/nething.error

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

export NCCL_GAUGE_HOME="/home/ldai8/ccl/netgauge-test/nething"

export NCCL_DEBUG="WARN"
export NCCL_PROTO="Simple"

NETHING_HOME="$NCCL_GAUGE_HOME/polaris"

export GAUGE_OUT_DIRE=$NETHING_HOME
export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"

export ITERATION_TIME="1"

export COMM_GPU_ID="0"

export GAUGE_MIN_NTHREADS=64
export GAUGE_MAX_NTHREADS=64

export GAUGE_MIN_NCHANNELS=1
export GAUGE_MAX_NCHANNELS=1

GAUGE_STEP_SIZE_SMALL=32
GAUGE_STEP_SIZE_MEDIUM=64
GAUGE_STEP_SIZE_LARGE=128

# when test G for small message:
# ch2 MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL / 16)) 
# ch1 MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL / 16))
# else, GAUGE_STEP_SIZE_SMALL 

if [ "$GAUGE_MAX_NCHANNELS" -eq 2 ]; then

    MESSAGE_SIZE_SMALL_START=$((GAUGE_STEP_SIZE_SMALL))
    MESSAGE_SIZE_SMALL_END=$((GAUGE_STEP_SIZE_SMALL * 8))
    MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL))

    MESSAGE_SIZE_MEDIUM_START=$((GAUGE_STEP_SIZE_MEDIUM * 4))
    MESSAGE_SIZE_MEDIUM_END=$((GAUGE_STEP_SIZE_MEDIUM * 32))
    MESSAGE_SIZE_MEDIUM_STEP=$((GAUGE_STEP_SIZE_MEDIUM * 2))

    MESSAGE_SIZE_LARGE_START=$((GAUGE_STEP_SIZE_LARGE * 16))
    MESSAGE_SIZE_LARGE_END=$((GAUGE_STEP_SIZE_LARGE * 128))
    MESSAGE_SIZE_LARGE_STEP=$((GAUGE_STEP_SIZE_LARGE * 8))

elif [ "$GAUGE_MAX_NCHANNELS" -eq 1 ]; then

    MESSAGE_SIZE_SMALL_START=$((GAUGE_STEP_SIZE_SMALL))
    MESSAGE_SIZE_SMALL_END=$((GAUGE_STEP_SIZE_SMALL * 4))
    MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL))

    MESSAGE_SIZE_MEDIUM_START=$((GAUGE_STEP_SIZE_MEDIUM * 2))
    MESSAGE_SIZE_MEDIUM_END=$((GAUGE_STEP_SIZE_MEDIUM * 16))
    MESSAGE_SIZE_MEDIUM_STEP=$((GAUGE_STEP_SIZE_MEDIUM * 1))

    MESSAGE_SIZE_LARGE_START=$((GAUGE_STEP_SIZE_LARGE * 8))
    MESSAGE_SIZE_LARGE_END=$((GAUGE_STEP_SIZE_LARGE * 128))
    MESSAGE_SIZE_LARGE_STEP=$((GAUGE_STEP_SIZE_LARGE * 8))

fi

# Read the list of allocated nodes
nodes=($(sort -u $PBS_NODEFILE))

export TMPDIR="$NETHING_HOME/tmp"

cd $NETHING_HOME


# benchmarks for G g o L

# /home1/09168/ldai1/bin/dool --time --mem --cpu --net -N eno1,ib0,lo,total 1 > $NCCL_GAUGE_HOME/frontera/dool.csv &

for ((itr = 0; itr < ${ITERATION_TIME}; itr += 1)); do

    # NCCL source location
    NCCL_SRC_LOCATION="/home/ldai8/ccl/NCCL_profile"

    # Update to include the correct path for NVCC and MPI library paths
    export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

    # for sync_mode in sync group; do
    for sync_mode in sync; do
        for ((n = 10; n <= 10; n *= 10)); do
            for ((nch = ${GAUGE_MIN_NCHANNELS}; nch <= ${GAUGE_MAX_NCHANNELS}; nch *= 2)); do
                for mode in pping; do
                    for ((nth = ${GAUGE_MIN_NTHREADS}; nth <= ${GAUGE_MAX_NTHREADS}; nth *= 2)); do
                        for ((d = 0; d <= 0; d += 1)); do
                            export GAUGE_NCHANNELS=${nch}
                            export GAUGE_MODE=${mode}
                            export NCCL_MIN_NCHANNELS=${nch}
                            export NCCL_MAX_NCHANNELS=${nch}
                            export GAUGE_MESSAGE_SIZE=1
                            export NCCL_NTHREADS=${nth}
                            export GAUGE_STEP_SIZE="0"
                            # ####################################################### 1 byte message size ###############################
                            # e=0
                            # start=$((e * 2))
                            # node_list="${nodes[start]},${nodes[start + 1]}"
                            # export GAUGE_EXPERIMENT_ID=${e}
                            # export GAUGE_ITERATION=$((itr + e * 100))                            
                            # $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --hosts $node_list \
                            # --cpu-bind core nsys profile -t cuda,nvtx,osrt,cublas,mpi --mpi-impl=mpich \
                            # --stats=true -o $NETHING_HOME/nsys_result/ncclNething_%q{PMI_RANK}_${GAUGE_MESSAGE_SIZE}  --gpu-metrics-device=all --cuda-memory-usage=true --export=sqlite \
                            # $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}_e_${e}.exe    
                            # ####################################################### small message size ###############################
                            # export GAUGE_STEP_SIZE="32"
                            # for ((msize=${MESSAGE_SIZE_SMALL_START}; msize<${MESSAGE_SIZE_SMALL_END}; msize+=${MESSAGE_SIZE_SMALL_STEP})); do
                            #     # Run 5 instances, each using 2 nodes
                            #     e=0
                            #     start=$((e * 2))
                            #     node_list="${nodes[start]},${nodes[start + 1]}"
                            #     export GAUGE_EXPERIMENT_ID=${e}
                            #     export GAUGE_ITERATION=$((itr + e * 100))
                            #     export GAUGE_MESSAGE_SIZE=${msize}
                            #     # $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --hosts $node_list --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}_e_${e}.exe
                            #     $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --hosts $node_list \
                            #     --cpu-bind core nsys profile -t cuda,nvtx,osrt,cublas,mpi --mpi-impl=mpich \
                            #     --stats=true -o $NETHING_HOME/nsys_result/ncclNething_%q{PMI_RANK}_${msize}  --gpu-metrics-device=all --cuda-memory-usage=true --export=sqlite \
                            #     $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}_e_${e}.exe 
                            # done
                            # ####################################################### medium message size ###############################
                            # export GAUGE_STEP_SIZE="64"
                            # for ((msize=${MESSAGE_SIZE_MEDIUM_START}; msize<${MESSAGE_SIZE_MEDIUM_END}; msize+=${MESSAGE_SIZE_MEDIUM_STEP})); do
                            #     # Run 5 instances, each using 2 nodes
                            #     e=0
                            #     start=$((e * 2))
                            #     node_list="${nodes[start]},${nodes[start + 1]}"
                            #     export GAUGE_EXPERIMENT_ID=${e}
                            #     export GAUGE_ITERATION=$((itr + e * 100))
                            #     export GAUGE_MESSAGE_SIZE=${msize}
                            #     $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --hosts $node_list \
                            #     --cpu-bind core nsys profile -t cuda,nvtx,osrt,cublas,mpi --mpi-impl=mpich \
                            #     --stats=true -o $NETHING_HOME/nsys_result/ncclNething_%q{PMI_RANK}_${msize}  --gpu-metrics-device=all --cuda-memory-usage=true --export=sqlite \
                            #     $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}_e_${e}.exe 
                            # done
                            # ####################################################### large message size ###############################
                            # export GAUGE_STEP_SIZE="128"
                            # for ((msize=${MESSAGE_SIZE_LARGE_START}; msize<=${MESSAGE_SIZE_LARGE_END}; msize+=${MESSAGE_SIZE_LARGE_STEP})); do
                            #     # Run 5 instances, each using 2 nodes
                            #         e=0
                            #         start=$((e * 2))
                            #         node_list="${nodes[start]},${nodes[start + 1]}"
                            #         export GAUGE_EXPERIMENT_ID=${e}
                            #         export GAUGE_ITERATION=$((itr + e * 100))
                            #         export GAUGE_MESSAGE_SIZE=${msize}
                            #         $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --hosts $node_list \
                            #         --cpu-bind core nsys profile -t cuda,nvtx,osrt,cublas,mpi --mpi-impl=mpich \
                            #         --stats=true -o $NETHING_HOME/nsys_result/ncclNething_%q{PMI_RANK}_${msize}  --gpu-metrics-device=all --cuda-memory-usage=true --export=sqlite \
                            #         $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}_e_${e}.exe 
                            # done
                            ###################################################### ext message size ###############################
                            export GAUGE_STEP_SIZE="1048576"
                            for ((msize=(${GAUGE_STEP_SIZE}); msize<=16*${GAUGE_STEP_SIZE}; msize+=${GAUGE_STEP_SIZE})); do
                                # Run 5 instances, each using 2 nodes
                                e=0
                                start=$((e * 2))
                                node_list="${nodes[start]},${nodes[start + 1]}"
                                export GAUGE_EXPERIMENT_ID=${e}
                                export GAUGE_ITERATION=$((itr + e * 100))
                                export GAUGE_MESSAGE_SIZE=${msize}
                                $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --hosts $node_list \
                                --cpu-bind core nsys profile -t cuda,nvtx,osrt,cublas,mpi --mpi-impl=mpich \
                                --stats=true -o $NETHING_HOME/nsys_result/ncclNething_%q{PMI_RANK}_${msize}  --gpu-metrics-device=all --cuda-memory-usage=true --export=sqlite \
                                $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}_e_${e}.exe 
                            done
                        done
                    done
                done
            done 
        done
    done



    # # NCCL source location
    # NCCL_SRC_LOCATION="/home/ldai8/ccl/NCCL_profile_D"

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
    #                         export GAUGE_STEP_SIZE="0"
    #                          # Run 5 instances, each using 2 nodes
    #                         for e in {0}; do
    #                             start=$((e * 2))
    #                             node_list="${nodes[start]},${nodes[start + 1]}"
    #                             export GAUGE_EXPERIMENT_ID=${e}
    #                             export GAUGE_ITERATION=$((itr + e * 100))
    #                             $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --hosts $node_list --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}_e_${e}.exe &
    #                         done
    #                         wait
    #                         export GAUGE_STEP_SIZE="32"
    #                         for ((msize=${MESSAGE_SIZE_SMALL_START}; msize<${MESSAGE_SIZE_SMALL_END}; msize+=${MESSAGE_SIZE_SMALL_STEP})); do
    #                             # Run 5 instances, each using 2 nodes
    #                             for e in {0}; do
    #                                 start=$((e * 2))
    #                                 node_list="${nodes[start]},${nodes[start + 1]}"
    #                                 export GAUGE_EXPERIMENT_ID=${e}
    #                                 export GAUGE_ITERATION=$((itr + e * 100))
    #                                 export GAUGE_MESSAGE_SIZE=${msize}
    #                                 $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --hosts $node_list --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}_e_${e}.exe &
    #                             done
    #                             wait
    #                         done
    #                         export GAUGE_STEP_SIZE="64"
    #                         for ((msize=${MESSAGE_SIZE_MEDIUM_START}; msize<${MESSAGE_SIZE_MEDIUM_END}; msize+=${MESSAGE_SIZE_MEDIUM_STEP})); do
    #                             # Run 5 instances, each using 2 nodes
    #                             for e in {0}; do
    #                                 start=$((e * 2))
    #                                 node_list="${nodes[start]},${nodes[start + 1]}"
    #                                 export GAUGE_EXPERIMENT_ID=${e}
    #                                 export GAUGE_ITERATION=$((itr + e * 100))
    #                                 export GAUGE_MESSAGE_SIZE=${msize}
    #                                 $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --hosts $node_list --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}_e_${e}.exe &
    #                             done
    #                             wait
    #                         done
    #                         export GAUGE_STEP_SIZE="128"
    #                         for ((msize=${MESSAGE_SIZE_LARGE_START}; msize<=${MESSAGE_SIZE_LARGE_END}; msize+=${MESSAGE_SIZE_LARGE_STEP})); do
    #                             # Run 5 instances, each using 2 nodes
    #                             for e in {0}; do
    #                                 start=$((e * 2))
    #                                 node_list="${nodes[start]},${nodes[start + 1]}"
    #                                 export GAUGE_EXPERIMENT_ID=${e}
    #                                 export GAUGE_ITERATION=$((itr + e * 100))
    #                                 export GAUGE_MESSAGE_SIZE=${msize}
    #                                 $MPIEXEC_HOME/bin/mpirun -n 2 --ppn 1 --hosts $node_list --cpu-bind core $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_${sync_mode}_d_${d}_e_${e}.exe &
    #                             done
    #                             wait
    #                         done
    #                     done
    #                 done
    #             done
    #         done 
    #     done
    # done

done

# kill %1

if [ -d "./tmp" ]; then
    rm -rf ./tmp/*
else
    echo "./tmp directory does not exist."
fi





