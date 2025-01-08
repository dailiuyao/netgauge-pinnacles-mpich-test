#!/bin/bash
#SBATCH -A m4753 
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH -t 00:09:59        # Run time (hh:mm:ss)
#SBATCH -J ncclgauge           # Job name
#SBATCH -o ./log/ncclgauge.o%j       # Name of stdout output file
#SBATCH -e ./log/ncclgauge.e%j       # Name of stderr error file
#SBATCH --gpu-bind=none

# Set environment variables

module load cudatoolkit

export MPICH_GPU_SUPPORT_ENABLED=1

export FI_LOG_LEVEL=info
export FI_LOG_PROV=all

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2
export MPI_HOME=/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3

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

# export MPIEXEC_HOME=/opt/cray/pals/1.3.4
export NCCL_NET_PLUGIN_HOME="/global/homes/l/ldai8/ccl/aws-ofi-nccl-1.13.2-aws/build"     
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export FI_PROVIDER=cxi
# export NCCL_IB_HCA=cxi0,cxi1
export LD_LIBRARY_PATH=${NCCL_NET_PLUGIN_HOME}/lib:$LD_LIBRARY_PATH

# Additional compiler flags for NVCC
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

export NCCL_GAUGE_HOME="/global/homes/l/ldai8/ccl/netgauge-test/ncclguage"

export NCCL_DEBUG="INFO"
export NCCL_PROTO="Simple"

cd $NCCL_GAUGE_HOME/perlmutter

export GAUGE_OUT_DIRE="$NCCL_GAUGE_HOME/perlmutter/out"
export GAUGE_HEO="inter"
export GAUGE_CHUNK_SIZE="2"
export COMM_GPU_ID="0"

GAUGE_MIN_NTHREADS=64
GAUGE_MAX_NTHREADS=64

GAUGE_MIN_NCHANNELS=1
GAUGE_MAX_NCHANNELS=1

GAUGE_STEP_SIZE_SMALL=32
GAUGE_STEP_SIZE_MEDIUM=64
GAUGE_STEP_SIZE_LARGE=128

# when test G for small message:
# ch2 MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL / 16)) 
# ch1 MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL / 16))
# else, GAUGE_STEP_SIZE_SMALL 

if [ "$GAUGE_MAX_NCHANNELS" -eq 2 ]; then

    MESSAGE_SIZE_SMALL_START=$((GAUGE_STEP_SIZE_SMALL))
    MESSAGE_SIZE_SMALL_END=$((GAUGE_STEP_SIZE_SMALL * 7))
    MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL))

    MESSAGE_SIZE_MEDIUM_START=$((GAUGE_STEP_SIZE_MEDIUM * 4))
    MESSAGE_SIZE_MEDIUM_END=$((GAUGE_STEP_SIZE_MEDIUM * 28))
    MESSAGE_SIZE_MEDIUM_STEP=$((GAUGE_STEP_SIZE_MEDIUM * 4))

    MESSAGE_SIZE_LARGE_START=$((GAUGE_STEP_SIZE_LARGE * 16))
    MESSAGE_SIZE_LARGE_END=$((GAUGE_STEP_SIZE_LARGE * 128))
    MESSAGE_SIZE_LARGE_STEP=$((GAUGE_STEP_SIZE_LARGE * 16))

elif [ "$GAUGE_MAX_NCHANNELS" -eq 1 ]; then

    MESSAGE_SIZE_SMALL_START=$((GAUGE_STEP_SIZE_SMALL))
    MESSAGE_SIZE_SMALL_END=$((GAUGE_STEP_SIZE_SMALL * 3))
    MESSAGE_SIZE_SMALL_STEP=$((GAUGE_STEP_SIZE_SMALL))

    MESSAGE_SIZE_MEDIUM_START=$((GAUGE_STEP_SIZE_MEDIUM * 4))
    MESSAGE_SIZE_MEDIUM_END=$((GAUGE_STEP_SIZE_MEDIUM * 4))
    MESSAGE_SIZE_MEDIUM_STEP=$((GAUGE_STEP_SIZE_MEDIUM * 4))

    MESSAGE_SIZE_LARGE_START=$((GAUGE_STEP_SIZE_LARGE * 16))
    MESSAGE_SIZE_LARGE_END=$((GAUGE_STEP_SIZE_LARGE * 128))
    MESSAGE_SIZE_LARGE_STEP=$((GAUGE_STEP_SIZE_LARGE * 16))

fi

MESSAGE_SIZE_EXTRA_START=65536
MESSAGE_SIZE_EXTRA_END=524288
MESSAGE_SIZE_EXTRA_STEP=65536

# Read the list of allocated nodes
message_number=(1 2 4 8 16)
test_mode=("pping")
instance_number=1
instance_itr_number=1
instance_itr_start=0

# benchmarks for G g o L
# nvidia-smi --query-gpu=index,name,clocks.gr --format=csv -l 1 > $GAUGE_OUT_DIRE/gpu_log.csv &

# sh $GAUGE_OUT_DIRE/cpu_fre.sh &

# watch -n 1 -t "lscpu | grep 'MHz' >> $GAUGE_OUT_DIRE/cpu_frequency_log.txt" &

# /home1/09168/ldai1/bin/dool --time --mem --cpu --net -N eno1,ib0,lo,total 1 > $NCCL_GAUGE_HOME/frontera/dool.csv &

export NCCL_SRC_LOCATION=/global/homes/l/ldai8/ccl/NCCL_profile 

run_experiment() {

    # Update paths
    export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}
    export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${MPI_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

    for n in "${message_number[@]}"; do
        for ((nch = GAUGE_MIN_NCHANNELS; nch <= GAUGE_MAX_NCHANNELS; nch *= 2)); do
            for mode in "${test_mode[@]}"; do
                for ((nth = GAUGE_MIN_NTHREADS; nth <= GAUGE_MAX_NTHREADS; nth *= 2)); do
                    for d in "${d_number[@]}"; do
                        export GAUGE_NCHANNELS=${nch}
                        export GAUGE_MODE=${mode}
                        export NCCL_MIN_NCHANNELS=${nch}
                        export NCCL_MAX_NCHANNELS=${nch}
                        export NCCL_NTHREADS=${nth}
                        
                        # export GAUGE_STEP_SIZE="0"
                        # export GAUGE_MESSAGE_SIZE=1
                        # # Run experiments
                        # for ((e=0; e < instance_number; e += 1)); do
                        #     start=$((e * 2))
                        #     export GAUGE_EXPERIMENT_ID=${e}
                        #     export GAUGE_ITERATION=$((itr + e * instance_itr_number))
                        #     srun $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_e_${e}.exe ${d} &
                        # done
                        # wait

                        for GAUGE_STEP_SIZE in  "32" "64" "128" "524288"; do
                            export GAUGE_STEP_SIZE
                            case $GAUGE_STEP_SIZE in
                                "64")
                                    START_VAR="MESSAGE_SIZE_MEDIUM_START"
                                    END_VAR="MESSAGE_SIZE_MEDIUM_END"
                                    STEP_VAR="MESSAGE_SIZE_MEDIUM_STEP"
                                    ;;
                                "128")
                                    START_VAR="MESSAGE_SIZE_LARGE_START"
                                    END_VAR="MESSAGE_SIZE_LARGE_END"
                                    STEP_VAR="MESSAGE_SIZE_LARGE_STEP"
                                    ;;
                                "524288")
                                    START_VAR="MESSAGE_SIZE_EXTRA_START"
                                    END_VAR="MESSAGE_SIZE_EXTRA_END"
                                    STEP_VAR="MESSAGE_SIZE_EXTRA_STEP"
                                    ;;
                                *)
                                    START_VAR="MESSAGE_SIZE_SMALL_START"
                                    END_VAR="MESSAGE_SIZE_SMALL_END"
                                    STEP_VAR="MESSAGE_SIZE_SMALL_STEP"
                                    ;;
                            esac

                            for ((msize=${!START_VAR}; msize<=${!END_VAR}; msize+=${!STEP_VAR})); do
                                for ((e=0; e < instance_number; e++)); do
                                    start=$((e * 2))
                                    export GAUGE_EXPERIMENT_ID=${e} GAUGE_ITERATION=$((itr + e * instance_itr_number)) GAUGE_MESSAGE_SIZE=${msize}
                                    srun $NCCL_GAUGE_HOME/gauge/${mode}_gauge_n_${n}_e_${e}.exe ${d} &
                                done
                                wait
                            done
                        done
                    done
                done
            done
        done
    done
}

# Main loop
for ((itr = instance_itr_start; itr < ${instance_itr_number}; itr += 1)); do
    d_number=(0 200000)
    run_experiment
done

# kill %1
# kill %2
