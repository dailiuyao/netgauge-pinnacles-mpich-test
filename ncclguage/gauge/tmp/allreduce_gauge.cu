#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

struct LogMessage_lyd_global* d_messages;
// int nccl_gauge_iteration = 0;

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


int main(int argc, char* argv[])
{

  const char* env_gauge_heo_var = getenv("GAUGE_HEO");

  const char* env_gauge_mode_var = getenv("GAUGE_MODE");

  const char* env_gauge_iteration_var = getenv("GAUGE_ITERATION");

  const char* env_gauge_nchannels_var = getenv("GAUGE_NCHANNELS");

  const char* env_gauge_chunk_size_var = getenv("GAUGE_CHUNK_SIZE");

  const char* env_gauge_output_dir_var = getenv("GAUGE_OUT_DIRE");

  // Check if environment variables are set
  if (!env_gauge_heo_var) env_gauge_heo_var = "unknown_gauge_heo";
  if (!env_gauge_mode_var) env_gauge_mode_var = "unknown_gauge_mode";
  if (!env_gauge_iteration_var) env_gauge_iteration_var = "unknown_gauge_iteration";
  if (!env_gauge_nchannels_var) env_gauge_nchannels_var = "unknown_gauge_nchannels";
  if (!env_gauge_chunk_size_var) env_gauge_chunk_size_var = "unknown_gauge_chunk_size";
  if (!env_gauge_output_dir_var) {
    env_gauge_output_dir_var = "unknown_gauge_output_dir";
    printf("unknown gauge output dir\n");
  }


  int size = 1;  // Default size
  const char* env_gauge_size_var = getenv("GAUGE_MESSAGE_SIZE");
  if (env_gauge_size_var != nullptr) {
      size = atoi(env_gauge_size_var) * 1024 / 4;  // Convert from kilobytes to number of floats, assuming the environment variable is in kilobytes
  }


  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  char filename[256];

  if (myRank < 2) {
    sprintf(filename, "%s/nccl_pping_%s_chunk%s_itr0-5-r%d.out", env_gauge_output_dir_var, env_gauge_heo_var, env_gauge_chunk_size_var, myRank);
    freopen(filename, "a", stdout);
  } else {
    freopen("/dev/null", "w", stdout);
  }


  // int nccl_start = 0;
  // int nccl_end = 0;

  // nccl_start = clock();

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


  ncclUniqueId id;
  ncclComm_t comm;
  float *sendbuff, *recvbuff;
  cudaStream_t s;


  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaStreamCreate(&s));
  

  //gauge test
  CUDACHECK(cudaMalloc(&d_messages, sizeof(LogMessage_lyd_global)));
  CUDACHECK(cudaMemset(d_messages, 0, sizeof(LogMessage_lyd_global))); 

  // // Declare CUDA events
  // cudaEvent_t start_0, stop_0;
  // cudaEventCreate(&start_0);
  // cudaEventCreate(&stop_0);
  // float milliseconds_0 = 0;

  // cudaEventRecord(start_0, s);

  uint64_t kernel_gauge_start = rdtsc();


  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  //communicating using NCCL
  NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
        comm, s));

  // cudaEventRecord(stop_0, s);

  // cudaEventSynchronize(stop_0);

  // cudaEventElapsedTime(&milliseconds_0, start_0, stop_0);

  // printf("heo(%s)_mode(%s)_nchannels(%s)_chunk size(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_chunk_size_var, env_gauge_size_var, N_ITERS, env_gauge_iteration_var, milliseconds_0/1.44e3);


  // // Clean up
  // cudaEventDestroy(start_0);
  // cudaEventDestroy(stop_0);


  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));

  uint64_t kernel_gauge_end = rdtsc();

  // After the kernel execution, copy the messages back to the host
  LogMessage_lyd_global* h_messages = new LogMessage_lyd_global;
  cudaMemcpy(h_messages, d_messages, sizeof(LogMessage_lyd_global), cudaMemcpyDeviceToHost);

  // Process and print the messages on the host
  #if PROFILE_LYD_REDUCE_BROADCAST == 1
  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | allreduce.h | runTreeUpDown | recvReduceCopy | time: %f us\n", h_messages->timeValue[i][0]);
  }

  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | allreduce.h | runTreeUpDown | directSendFromOutput | time: %f us\n", h_messages->timeValue1[i][0]);
  }
  #endif

  #if PROFILE_LYD_REDUCE_BROADCAST_CHUNK == 1
  for (size_t i = 0; i < maxMessages; ++i) {
    for (size_t j = 0; j < MAXLOGLYD; j++){
      printf("DEVICE | allreduce.h | runTreeUpDown | recvReduceCopy-chunk | iteration %d | time: %f us\n", j, h_messages->timeValue[i][j]);
    }
  }
  for (size_t i = 0; i < maxMessages; ++i) {
    for (size_t j = 0; j < MAXLOGLYD; j++){
      printf("DEVICE | allreduce.h | runTreeUpDown | directSendFromOutput-chunk | iteration %d | time: %f us\n", j, h_messages->timeValue1[i][j]);
    }
  }
  #endif

  #if PROFILE_LYD_REDUCE_LOADCONN_SETDATA == 1
  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | prims_simple.h | primitives | loadRecvConn | time: %f us\n", h_messages->timeValue[i][0]);
  }

  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | prims_simple.h | primitives | loadSendConn | time: %f us\n", h_messages->timeValue1[i][0]);
  }

  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | prims_simple.h | primitives | setDataPtrs | time: %f us\n", h_messages->timeValue2[i][0]);
  }
  #endif

  #if PROFILE_LYD_GENERIC == 1
  for (size_t i = 0; i < maxMessages; ++i) {
    printf("DEVICE | prims_simple.h | genericop | time: %f us\n", h_messages->timeValue[i][0]);
  }
  #endif

  #if PROFILE_LYD_WAIT_REDUCE_COPY_POST == 1
  for (size_t i = 0; i < maxMessages; ++i) {
    for (size_t j = 0; j < MAXLOGLYD; j++){
      printf("DEVICE | prims_simple.h | genericOp | waitpeer | iteration %d | time: %f us\n", j, h_messages->timeValue[i][j]);
    }
  }
  for (size_t i = 0; i < maxMessages; ++i) {
    for (size_t j = 0; j < MAXLOGLYD; j++){
      printf("DEVICE | prims_simple.h | genericOp | ReduceOrCopyMulti | iteration %d | time: %f us\n", j, h_messages->timeValue1[i][j]);
    }
  }
  for (size_t i = 0; i < maxMessages; ++i) {
    for (size_t j = 0; j < MAXLOGLYD; j++){
      printf("DEVICE | prims_simple.h | genericOp | postPeer | iteration %d | time: %f us\n", j, h_messages->timeValue2[i][j]);
    }
  }
  #endif

  #if PROFILE_LYD_SEND_RECV_CHUNK == 1
  double gauge_time;

  if (myRank == 0) { 
    gauge_time = static_cast<double>(h_messages->timeValue[1][0] - h_messages->timeValue[0][0]) / 1410.0;
    printf("heo(%s)_mode(%s)_nchannels(%s)_chunk size(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_chunk_size_var, env_gauge_size_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, gauge_time);
    printf("nccl kernel elapsed time: %f us\n", static_cast<double>(kernel_gauge_end - kernel_gauge_start) / 2800.0);
  } else {
    gauge_time = static_cast<double>(h_messages->timeValue[0][0] - h_messages->timeValue[1][0]) / 1410.0; 
    printf("heo(%s)_mode(%s)_nchannels(%s)_chunk size(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_chunk_size_var, env_gauge_size_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, gauge_time);
  }

  // print the gap between chunks
  if (myRank == 0) { 
    for (size_t i = 1; i < N_ITERS; ++i) {
      gauge_time = static_cast<double>(h_messages->timeValue[0][i] - h_messages->timeValue[0][0]) / 1410.0;
      printf("chunk gap | chunk%d -> chunk0 | heo(%s)_mode(%s)_nchannels(%s)_chunk size(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f us\n", i, env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_chunk_size_var, env_gauge_size_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, gauge_time);
    }
  } else {
    for (size_t i = 1; i < N_ITERS; ++i) {
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[1][0]) / 1410.0; 
      printf("chunk gap | chunk%d -> chunk0 | heo(%s)_mode(%s)_nchannels(%s)_chunk size(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f us\n", i, env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_chunk_size_var, env_gauge_size_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, gauge_time);
    }
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

  // nccl_end = clock();

  // printf("heo(%s)_mode(%s)_nchannels(%s)_chunk size(%s)_message size(%s)_n(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_chunk_size_var, env_gauge_size_var, N_ITERS, env_gauge_iteration_var, (nccl_end - nccl_start)/1.44e3);

  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}