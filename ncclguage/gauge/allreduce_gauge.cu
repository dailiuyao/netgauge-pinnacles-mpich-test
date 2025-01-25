#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

struct LogMessage_lyd* d_messages;

std::chrono::time_point<std::chrono::high_resolution_clock> netIsend_time_start[MAXLOGLYD];
std::chrono::time_point<std::chrono::high_resolution_clock> netIrecv_time_start[MAXLOGLYD];

// #define WARMUP_ITERATION 5
#define WARMUP_SIZE 32
#define DEFAULT_D 0

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

uint64_t rdtsc() {
    uint32_t lo, hi;
    // Inline assembly to read the TSC
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (uint64_t)hi << 32 | lo;
}

void busyWaitMilliseconds(int ms) {
    auto start = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::milliseconds(ms);

    // Spin in a loop until the desired time has elapsed
    while (std::chrono::high_resolution_clock::now() - start < duration) {
        // Do nothing
    }
}



int main(int argc, char* argv[])
{

  const char* env_gauge_heo_var = getenv("GAUGE_HEO");

  const char* env_gauge_mode_var = getenv("GAUGE_MODE");

  const char* env_gauge_iteration_var = getenv("GAUGE_ITERATION");

  const char* env_gauge_nchannels_var = getenv("GAUGE_NCHANNELS");

  const char* env_gauge_chunk_size_var = getenv("GAUGE_CHUNK_SIZE");

  const char* env_gauge_output_dir_var = getenv("GAUGE_OUT_DIRE");

  const char* env_gauge_nthreads_var = getenv("NCCL_NTHREADS");

  const char* env_comm_gpu_id_var = getenv("COMM_GPU_ID");

  const char* env_experiment_id_var = getenv("GAUGE_EXPERIMENT_ID");

  const char* env_gauge_algo_var = getenv("NCCL_ALGO");


  // Check if environment variables are set
  if (!env_gauge_heo_var) env_gauge_heo_var = "unknown_gauge_heo";
  if (!env_gauge_mode_var) env_gauge_mode_var = "unknown_gauge_mode";
  if (!env_gauge_iteration_var) env_gauge_iteration_var = "unknown_gauge_iteration";
  if (!env_gauge_nchannels_var) env_gauge_nchannels_var = "unknown_gauge_nchannels";
  if (!env_gauge_chunk_size_var) env_gauge_chunk_size_var = "unknown_gauge_chunk_size";
  if (!env_gauge_nthreads_var) env_gauge_nthreads_var = "unknown_gauge_nthreads";  
  if (!env_gauge_output_dir_var) {
    env_gauge_output_dir_var = "unknown_gauge_output_dir";
    printf("unknown gauge output dir\n");
  }

  long long size = 1;  // Default size
  const char* env_gauge_size_var = getenv("GAUGE_MESSAGE_SIZE");
  if (env_gauge_size_var != nullptr) {
      size = atoll(env_gauge_size_var) * 1024 / 4;  // Convert from kilobytes to number of floats, assuming the environment variable is in kilobytes
  }

  const char* env_gauge_step_var = getenv("GAUGE_STEP_SIZE");

  int gauge_step = atoi(env_gauge_step_var);

  int comm_gpu_id = atoi(env_comm_gpu_id_var);

  int gauge_d = DEFAULT_D;

  gauge_d = atoi(argv[1]);

  int N_CHUNKS;

  // if (gauge_step != 0) {
  //   if (gauge_step >= 16384) {
  //     N_CHUNKS = 128;
  //   } else {
  //     N_CHUNKS = atoi(env_gauge_size_var)/atoi(env_gauge_step_var); 
  //   }
  // } else {
  //   N_CHUNKS = 1;
  // }

  // if (N_CHUNKS == 0) N_CHUNKS = 1;

  N_CHUNKS = MAXLOGLYD - 1;

  int myRank, nRanks, localRank = 0;

  // Set the device scheduling flag before creating a device context
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set device flags: %s\n", cudaGetErrorString(err));
        return 1;
    }

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  char filename[256];

  if (myRank == 0 || myRank == nRanks / 2 || myRank == nRanks / 2 - 1 || myRank == nRanks / 2 + 1) {
    sprintf(filename, "%s/nccl_allreduce_%s_chunk-%s_r-%d_e-%s.out", env_gauge_output_dir_var, env_gauge_heo_var, env_gauge_chunk_size_var, myRank, env_experiment_id_var);
    freopen(filename, "a", stdout);
  } else {
    freopen("/dev/null", "w", stdout);
  }

  // printf("proc is: %d\n", myRank);
  // int gdb = 1;
  // if (myRank == 0){
  //   gdb = 0;
  //   printf("proc is: %d, pid is %d\n", myRank, (int)getpid());
  // }
  // while(gdb == 0){
  //   printf("loop\n");
  //   sleep(10);
  // }

  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  // select gpu on each node
  if (comm_gpu_id == 0) {
    localRank = localRank + 1;
  } else if (comm_gpu_id == 1){
    if (myRank == 0) localRank = localRank + 1;
  } else if (comm_gpu_id == 2){
    if (myRank == 1) localRank = localRank + 1;
  } else if (comm_gpu_id == 3){
    localRank = localRank + 3;
  } 


  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;


  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, WARMUP_ITERATION * WARMUP_SIZE * sizeof(float) + N_ITERS * size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, WARMUP_ITERATION * WARMUP_SIZE * sizeof(float) + N_ITERS * size * sizeof(float))); 
  CUDACHECK(cudaStreamCreate(&s));
  

  //gauge test
  LogMessage_lyd host_messages;
  // Initialize other members of host_messages as needed
  memset(&host_messages, 0, sizeof(LogMessage_lyd)); // Zero-initialize the struct
  host_messages.gauge_d = gauge_d;
  host_messages.gauge_iteration = WARMUP_ITERATION+N_ITERS;

  CUDACHECK(cudaMalloc(&d_messages, sizeof(LogMessage_lyd)));

  CUDACHECK(cudaMemcpy(d_messages, &host_messages, sizeof(LogMessage_lyd), cudaMemcpyHostToDevice));

  
  ////////////////////////////// PROFILE_LYD_AllReduce_DEVICE: START //////////////////////////////
  
  #if PROFILE_LYD_REDUCE_BROADCAST_CHUNK == 1
  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  //communicating using NCCL

  cudaEvent_t start, stop;
  float elapsed_time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i=0; i < N_ITERS; i++){
    netIsend_time_start[i] = std::chrono::high_resolution_clock::now(); 
    netIrecv_time_start[i] = std::chrono::high_resolution_clock::now(); 
  }

  // Warm up START 
  CUDACHECK(cudaStreamSynchronize(s));

  for (int i = 0 ; i < WARMUP_ITERATION; i++) {
    NCCLCHECK(ncclAllReduce((const void*)((float*)sendbuff + i * WARMUP_SIZE), 
                        (void*)((float*)recvbuff + i * WARMUP_SIZE), 
                        WARMUP_SIZE, 
                        ncclFloat, 
                        ncclSum, 
                        comm, 
                        s));
  }

  CUDACHECK(cudaStreamSynchronize(s));

  // Warm up END

  cudaEventRecord(start, s);

  std::chrono::time_point<std::chrono::high_resolution_clock> nccl_func_start_time = std::chrono::high_resolution_clock::now(); 

  CUDACHECK(cudaStreamSynchronize(s));

  for (int i = 0; i < N_ITERS; i++) {
    NCCLCHECK(ncclAllReduce((const void*)((float*)sendbuff + i * size), 
                            (void*)((float*)recvbuff + i * size), 
                            size, 
                            ncclFloat, 
                            ncclSum, 
                            comm, 
                            s));
    // busyWaitMilliseconds(gauge_d); 
    CUDACHECK(cudaStreamSynchronize(s));
  }
  CUDACHECK(cudaStreamSynchronize(s));

  cudaEventRecord(stop, s);

  std::chrono::time_point<std::chrono::high_resolution_clock> nccl_func_end_time = std::chrono::high_resolution_clock::now();

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  // Calculate elapsed time between events
  cudaEventElapsedTime(&elapsed_time, start, stop);

  // Destroy events
  cudaEventDestroy(start);
  cudaEventDestroy(stop); 

  std::chrono::duration<float, std::milli> nccl_func_time = nccl_func_end_time - nccl_func_start_time; 

  #endif

  ////////////////////////////// PROFILE_LYD_AllReduce_DEVICE: END //////////////////////////////

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  // After the kernel execution, copy the messages back to the host
  LogMessage_lyd* h_messages = new LogMessage_lyd;
  cudaMemcpy(h_messages, d_messages, sizeof(LogMessage_lyd), cudaMemcpyDeviceToHost);

  #if PROFILE_LYD_REDUCE_BROADCAST_CHUNK == 1
  double gauge_time;
  
  // if (strcmp(env_gauge_algo_var, "Tree") == 0)
  if (myRank == 0) {
    printf("INFO: heo(%s)_mode(%s)_message size(%s)_nchannels(%s)_nthreads(%s)_nmessages(%d)_d(%d)_iteration(%s)\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, gauge_d, env_gauge_iteration_var);
    printf("--nccl allreduce elapsed time by cuda event: %f ms\n", elapsed_time);
    printf("--nccl allreduce elapsed time by clock: %.3f ms\n", nccl_func_time.count());
    for (size_t i = 0; i < N_ITERS; ++i) {
      gauge_time = static_cast<double>(h_messages->timeEndValue[0][i+WARMUP_ITERATION] - h_messages->timeStartValue[0][i+WARMUP_ITERATION]) / GAUGE_GPU_FREQUENCY;
      printf("--message %d recvReduceCopy time: %f us\n", i, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeEndValue[1][i+WARMUP_ITERATION] - h_messages->timeStartValue[1][i+WARMUP_ITERATION]) / GAUGE_GPU_FREQUENCY;
      printf("--message %d directSendFromOutput time: %f us\n", i, gauge_time);
    }
  } else if (myRank == nRanks / 2) {
    printf("INFO: heo(%s)_mode(%s)_message size(%s)_nchannels(%s)_nthreads(%s)_nmessages(%d)_d(%d)_iteration(%s)\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, gauge_d, env_gauge_iteration_var);
    printf("--nccl allreduce elapsed time by cuda event: %f ms\n", elapsed_time);
    printf("--nccl allreduce elapsed time by clock: %.3f ms\n", nccl_func_time.count());
    for (size_t i = 0; i < N_ITERS; ++i) {
      gauge_time = static_cast<double>(h_messages->timeEndValue[0][i+WARMUP_ITERATION] - h_messages->timeStartValue[0][i+WARMUP_ITERATION]) / GAUGE_GPU_FREQUENCY;
      printf("--message %d recvReduceSend time: %f us\n", i, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeEndValue[1][i+WARMUP_ITERATION] - h_messages->timeStartValue[1][i+WARMUP_ITERATION]) / GAUGE_GPU_FREQUENCY;
      printf("--message %d directRecvCopySend time: %f us\n", i, gauge_time);
    }
  } else if (myRank == nRanks / 2 - 1) {
    printf("INFO: heo(%s)_mode(%s)_message size(%s)_nchannels(%s)_nthreads(%s)_nmessages(%d)_d(%d)_iteration(%s)\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, gauge_d, env_gauge_iteration_var);
    printf("--nccl allreduce elapsed time by cuda event: %f ms\n", elapsed_time);
    printf("--nccl allreduce elapsed time by clock: %.3f ms\n", nccl_func_time.count());
    for (size_t i = 0; i < N_ITERS; ++i) {
      gauge_time = static_cast<double>(h_messages->timeEndValue[0][i+WARMUP_ITERATION] - h_messages->timeStartValue[0][i+WARMUP_ITERATION]) / GAUGE_GPU_FREQUENCY;
      printf("--message %d send time: %f us\n", i, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeEndValue[1][i+WARMUP_ITERATION] - h_messages->timeStartValue[1][i+WARMUP_ITERATION]) / GAUGE_GPU_FREQUENCY;
      printf("--message %d directRecv time: %f us\n", i, gauge_time);
    }
  } else if (myRank == nRanks / 2 + 1) {
    printf("INFO: heo(%s)_mode(%s)_message size(%s)_nchannels(%s)_nthreads(%s)_nmessages(%d)_d(%d)_iteration(%s)\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, gauge_d, env_gauge_iteration_var);
    printf("--nccl allreduce elapsed time by cuda event: %f ms\n", elapsed_time);
    printf("--nccl allreduce elapsed time by clock: %.3f ms\n", nccl_func_time.count());
    for (size_t i = 0; i < N_ITERS; ++i) {
      gauge_time = static_cast<double>(h_messages->timeEndValue[0][i+WARMUP_ITERATION] - h_messages->timeStartValue[0][i+WARMUP_ITERATION]) / GAUGE_GPU_FREQUENCY;
      printf("--message %d send time: %f us\n", i, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeEndValue[1][i+WARMUP_ITERATION] - h_messages->timeStartValue[1][i+WARMUP_ITERATION]) / GAUGE_GPU_FREQUENCY;
      printf("--message %d directRecv time: %f us\n", i, gauge_time);
    }
  }

  // print the gap between chunks
  if (myRank == 0) { 
    for (size_t i = 1; i < min(static_cast<size_t>(h_messages->signal[0]), static_cast<size_t>(N_CHUNKS)); ++i) {
      gauge_time = static_cast<double>(h_messages->timeValue[0][i] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
      printf("--chunk gap | chunk%d recvReduceCopy - chunk0 recvReduceCopy start: %f us\n", i, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[1][0]) / GAUGE_GPU_FREQUENCY;
      printf("--chunk gap | chunk%d directSendFromOutput - chunk0 directSendFromOutput: %f us\n", i, gauge_time);
    }
    gauge_time = static_cast<double>(h_messages->timeValue[0][N_CHUNKS] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
    printf("--chunk gap | chunk%d recvReduceCopy - chunk0 recvReduceCopy: %f us\n", h_messages->signal[0], gauge_time);
    gauge_time = static_cast<double>(h_messages->timeValue[1][N_CHUNKS] - h_messages->timeValue[1][0]) / GAUGE_GPU_FREQUENCY;
    printf("--chunk gap | chunk%d directSendFromOutput - chunk0 directSendFromOutput: %f us\n", h_messages->signal[0], gauge_time);
  } else if (myRank == nRanks / 2) {
    for (size_t i = 1; i < min(static_cast<size_t>(h_messages->signal[0]), static_cast<size_t>(N_CHUNKS)); ++i) {
      gauge_time = static_cast<double>(h_messages->timeValue[0][i] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
      printf("--chunk gap | chunk%d recvReduceSend - chunk0 recvReduceSend: %f us\n", i, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[1][0]) / GAUGE_GPU_FREQUENCY;
      printf("--chunk gap | chunk%d directRecvCopySend - chunk0 directRecvCopySend: %f us\n", i, gauge_time);
    }
    gauge_time = static_cast<double>(h_messages->timeValue[0][N_CHUNKS] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
    printf("--chunk gap | chunk%d recvReduceSend - chunk0 recvReduceSend: %f us\n", h_messages->signal[0], gauge_time);
    gauge_time = static_cast<double>(h_messages->timeValue[1][N_CHUNKS] - h_messages->timeValue[1][0]) / GAUGE_GPU_FREQUENCY;
    printf("--chunk gap | chunk%d directRecvCopySend - chunk0 directRecvCopySend: %f us\n", h_messages->signal[0], gauge_time);
  } else if (myRank == nRanks / 2 - 1) {
    for (size_t i = 1; i < min(static_cast<size_t>(h_messages->signal[0]), static_cast<size_t>(N_CHUNKS)); ++i) {
      gauge_time = static_cast<double>(h_messages->timeValue[0][i] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
      printf("--chunk gap | chunk%d send - chunk0 send: %f us\n", i, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[1][0]) / GAUGE_GPU_FREQUENCY;
      printf("--chunk gap | chunk%d directRecv - chunk0 directRecv: %f us\n", i, gauge_time);
    }
    gauge_time = static_cast<double>(h_messages->timeValue[0][N_CHUNKS] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
    printf("--chunk gap | chunk%d send - chunk0 send: %f us\n", h_messages->signal[0], gauge_time);
    gauge_time = static_cast<double>(h_messages->timeValue[1][N_CHUNKS] - h_messages->timeValue[1][0]) / GAUGE_GPU_FREQUENCY;
    printf("--chunk gap | chunk%d directRecv - chunk0 directRecv: %f us\n", h_messages->signal[0], gauge_time);
  } else if (myRank == nRanks / 2 + 1) {
    for (size_t i = 1; i < min(static_cast<size_t>(h_messages->signal[0]), static_cast<size_t>(N_CHUNKS)); ++i) {
      gauge_time = static_cast<double>(h_messages->timeValue[0][i] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
      printf("--chunk gap | chunk%d send - chunk0 send: %f us\n", i, gauge_time);
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[1][0]) / GAUGE_GPU_FREQUENCY;
      printf("--chunk gap | chunk%d directRecv - chunk0 directRecv: %f us\n", i, gauge_time);
    }
    gauge_time = static_cast<double>(h_messages->timeValue[0][N_CHUNKS] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
    printf("--chunk gap | chunk%d send - chunk0 send: %f us\n", h_messages->signal[0], gauge_time);
    gauge_time = static_cast<double>(h_messages->timeValue[1][N_CHUNKS] - h_messages->timeValue[1][0]) / GAUGE_GPU_FREQUENCY;
    printf("--chunk gap | chunk%d directRecv - chunk0 directRecv: %f us\n", h_messages->signal[0], gauge_time);
  }
  #endif

  // Free the device memory of the gauge test
  cudaFree(d_messages);
  delete[] h_messages;


  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));


  //finalizing NCCL
  ncclCommDestroy(comm);

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}