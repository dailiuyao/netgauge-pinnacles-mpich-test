#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#define N_STREAMS 4

struct LogMessage_lyd* d_messages;
float netIsend_time;
float netIrecv_time; 
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

  const char* env_gauge_nthreads_var = getenv("NCCL_NTHREADS");

  const char* env_comm_gpu_id_var = getenv("COMM_GPU_ID");

  const char* env_experiment_id_var = getenv("GAUGE_EXPERIMENT_ID");

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

  // const char* env_gauge_d_var = getenv("GAUGE_D");

  // int loggp_gauge_d; 

  // if (env_gauge_d_var != NULL) {
  //       // Convert it to an integer
  //       loggp_gauge_d = atoi(env_gauge_d_var);

  //       // Example usage
  //       printf("GAUGE_D is set to: %d\n", loggp_gauge_d);
  // } else {
  //       printf("GAUGE_D is not set.\n");
  // }
  
  // // Copy the value to the device global variable
  // cudaMemcpyToSymbol(device_loggp_gauge_d, &loggp_gauge_d, sizeof(int));

  long long size = 1;  // Default size
  const char* env_gauge_size_var = getenv("GAUGE_MESSAGE_SIZE");
  if (env_gauge_size_var != nullptr) {
      size = atoll(env_gauge_size_var) * 1024 / 4;  // Convert from kilobytes to number of floats, assuming the environment variable is in kilobytes
  }

  const char* env_gauge_step_var = getenv("GAUGE_STEP_SIZE");

  int gauge_step = atoi(env_gauge_step_var);

  int comm_gpu_id = atoi(env_comm_gpu_id_var);

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

  N_CHUNKS = 32;

  int myRank, nRanks, localRank = 0;


  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  char filename[256];

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

  // const char*  syncmode;

  // if (PROFILE_LYD_P2P_HOST_SYNC == 1) {
  //   syncmode = "sync";
  // } else if (PROFILE_LYD_P2P_HOST_GROUP == 1) {
  //   syncmode = "group";
  // } else {
  //   syncmode = "device";
  // }

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
  ncclComm_t comm[N_STREAMS];
  float *sendbuff, *recvbuff;
  // cudaStream_t s;


  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&sendbuff, N_STREAMS * size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, N_STREAMS * size * sizeof(float)));
  
  // CUDACHECK(cudaStreamCreate(&s));

  cudaStream_t streams[N_STREAMS];
  for (int i = 0; i < N_STREAMS; ++i) {
      CUDACHECK(cudaStreamCreate(&streams[i]));
  }
  

  //gauge test
  CUDACHECK(cudaMalloc(&d_messages, sizeof(LogMessage_lyd)));
  CUDACHECK(cudaMemset(d_messages, 0, sizeof(LogMessage_lyd)));

  ////////////////////////////// PROFILE_LYD_P2P_HOST_SYNC: START //////////////////////////////
  
  #if PROFILE_LYD_P2P_HOST_SYNC == 1
  // Declare CUDA events in host code
  cudaEvent_t p2p_time_stamp[N_ITERS+1];
  for (int i = 0; i < (N_ITERS+1); ++i) {
    cudaEventCreate(&p2p_time_stamp[i]); // Initialize each event individually
  }

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));


  //communicating using NCCL
  //P2P
  int recvPeer = (myRank-1+nRanks) % nRanks;
  int sendPeer = (myRank+1) % nRanks;

  if (myRank == 0) {
    usleep(10);
    
    cudaEventRecord(p2p_time_stamp[0], s);
    NCCLCHECK(ncclSend((const void*)sendbuff, size, ncclFloat, sendPeer, comm, s));
    CUDACHECK(cudaStreamSynchronize(s));
    // #if N_ITERS > 1 
    // usleep(loggp_gauge_d);
    // #endif

    #if N_ITERS > 1
    for (int i = 1 ; i < N_ITERS; i++) {
      usleep(loggp_gauge_d);
      cudaEventRecord(p2p_time_stamp[i], s);
      NCCLCHECK(ncclSend((const void*)sendbuff, size, ncclFloat, sendPeer, comm, s));
      CUDACHECK(cudaStreamSynchronize(s));
      // if (i != (N_ITERS-1)) usleep(loggp_gauge_d);
    }
    #endif

    NCCLCHECK(ncclRecv((void*)recvbuff, size, ncclFloat, recvPeer, comm, s));
    CUDACHECK(cudaStreamSynchronize(s));

    cudaEventRecord(p2p_time_stamp[N_ITERS], s);
  } else {
    NCCLCHECK(ncclRecv((void*)recvbuff, size, ncclFloat, recvPeer, comm, s));
    CUDACHECK(cudaStreamSynchronize(s));

    #if N_ITERS > 1
    for (int i = 1 ; i < N_ITERS; i++) {
      NCCLCHECK(ncclRecv((void*)recvbuff, size, ncclFloat, recvPeer, comm, s));
      CUDACHECK(cudaStreamSynchronize(s));
    }
    #endif

    NCCLCHECK(ncclSend((const void*)sendbuff, size, ncclFloat, sendPeer, comm, s));
    CUDACHECK(cudaStreamSynchronize(s));
  }

  #endif

  ////////////////////////////// PROFILE_LYD_P2P_HOST_SYNC: END //////////////////////////////

  ////////////////////////////// PROFILE_LYD_P2P_HOST_GROUP: START //////////////////////////////

  #if PROFILE_LYD_P2P_HOST_GROUP == 1

  // Declare CUDA events in host code
  cudaEvent_t p2p_time_stamp[N_ITERS+1];
  for (int i = 0; i < (N_ITERS+1); ++i) {
    cudaEventCreate(&p2p_time_stamp[i]); // Initialize each event individually
  }

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  //communicating using NCCL
  //P2P
  int recvPeer = (myRank-1+nRanks) % nRanks;
  int sendPeer = (myRank+1) % nRanks;

  cudaEventRecord(p2p_time_stamp[0], s);
  for (int i = 0 ; i < N_ITERS; i++) {
    NCCLCHECK(ncclGroupStart());
    if (myRank == 0) {
      NCCLCHECK(ncclSend((const void*)sendbuff, size, ncclFloat, sendPeer, comm, s));
    } else {
      NCCLCHECK(ncclRecv((void*)recvbuff, size, ncclFloat, recvPeer, comm, s));
    }
    NCCLCHECK(ncclGroupEnd());
    CUDACHECK(cudaStreamSynchronize(s));
    if (myRank == 0 && i != (N_ITERS-1)) usleep(loggp_gauge_d);
  }
  NCCLCHECK(ncclGroupStart());
  if (myRank == 1) {
    NCCLCHECK(ncclSend((const void*)sendbuff, size, ncclFloat, sendPeer, comm, s));
  } else {
    NCCLCHECK(ncclRecv((void*)recvbuff, size, ncclFloat, recvPeer, comm, s));
  }
  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(cudaStreamSynchronize(s));
  cudaEventRecord(p2p_time_stamp[N_ITERS], s);

  #endif

  ////////////////////////////// PROFILE_LYD_P2P_HOST_GROUP: END //////////////////////////////

  ////////////////////////////// PROFILE_LYD_P2P_DEVICE_SYNC: START //////////////////////////////
  
  #if PROFILE_LYD_P2P_HOST_SYNC != 1 && PROFILE_LYD_P2P_HOST_GROUP != 1

  //initializing NCCL
  // NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  for (int i = 0; i < N_STREAMS; ++i) {
    NCCLCHECK(ncclCommInitRank(&comm[i], nRanks, id, myRank));
  }

  //communicating using NCCL
  //P2P
  int recvPeer = (myRank-1+nRanks) % nRanks;
  int sendPeer = (myRank+1) % nRanks;

  cudaEvent_t start[N_STREAMS], stop[N_STREAMS];
  float elapsed_time[N_STREAMS];

  for (int i = 0; i < N_STREAMS; ++i) {
      CUDACHECK(cudaEventCreate(&start[i]));
      CUDACHECK(cudaEventCreate(&stop[i]));
  }

  netIsend_time = 0;
  netIrecv_time = 0;

  // CUDACHECK(cudaStreamSynchronize(s));

  // Synchronize all streams after the operations are queued
  for (int j = 0; j < N_STREAMS; ++j) {
      CUDACHECK(cudaStreamSynchronize(streams[j]));
  }
  
  for (int j = 0; j < N_STREAMS; ++j) {
      cudaEventRecord(start[j], streams[j]);
  }

  float nccl_func_start_time = clock();  

  // for (int i = 0 ; i < N_ITERS; i++) {
  //   // NCCLCHECK(ncclGroupStart());
  //   if (myRank == 0) {
  //     NCCLCHECK(ncclSend((const void*)sendbuff, size, ncclFloat, sendPeer, comm, s));
  //   } else {
  //     NCCLCHECK(ncclRecv((void*)recvbuff, size, ncclFloat, recvPeer, comm, s));
  //   }
  //   // NCCLCHECK(ncclGroupEnd());
  // }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < N_ITERS; ++i) {
      if (myRank == 0) {
          for (int j = 0; j < N_STREAMS; ++j) {
              NCCLCHECK(ncclSend((const void*)((float*)sendbuff + j * size), size, ncclFloat, sendPeer, comm[j], streams[j]));
          }
      } else {
          for (int j = 0; j < N_STREAMS; ++j) {
              NCCLCHECK(ncclRecv((void*)((float*)recvbuff + j * size), size, ncclFloat, recvPeer, comm[j], streams[j]));
          }
      }
  }
  NCCLCHECK(ncclGroupEnd());

  for (int j = 0; j < N_STREAMS; ++j) {
      CUDACHECK(cudaStreamSynchronize(streams[j]));
  }

  NCCLCHECK(ncclGroupStart());
  if (myRank == 1) {
      for (int j = 0; j < N_STREAMS; ++j) {
          NCCLCHECK(ncclSend((const void*)((float*)sendbuff + j * size), size, ncclFloat, sendPeer, comm[j], streams[j]));
      }
  } else {
      for (int j = 0; j < N_STREAMS; ++j) {
          NCCLCHECK(ncclRecv((void*)((float*)recvbuff + j * size), size, ncclFloat, recvPeer, comm[j], streams[j]));
      }
  }
  NCCLCHECK(ncclGroupEnd());



  // // NCCLCHECK(ncclGroupStart());
  // if (myRank == 1) {
  //   NCCLCHECK(ncclSend((const void*)sendbuff, size, ncclFloat, sendPeer, comm, s));
  // } else {
  //   NCCLCHECK(ncclRecv((void*)recvbuff, size, ncclFloat, recvPeer, comm, s));
  // }

  
  // NCCLCHECK(ncclGroupEnd());
  for (int j = 0; j < N_STREAMS; ++j) {
      CUDACHECK(cudaStreamSynchronize(streams[j]));
  }

  // cudaEventRecord(stop, s);

  for (int j = 0; j < N_STREAMS; ++j) {
      cudaEventRecord(stop[j], streams[j]);
  }

  for (int j = 0; j < N_STREAMS; ++j) {
      cudaEventSynchronize(stop[j]);
  }

  float nccl_func_end_time = clock();  

  // // Calculate elapsed time between events
  // cudaEventElapsedTime(&elapsed_time, start, stop);

  for (int j = 0; j < N_STREAMS; ++j) {
      cudaEventElapsedTime(&elapsed_time[j], start[j], stop[j]);
  }


  // // Destroy events
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop); 

  for (int j = 0; j < N_STREAMS; ++j) {
    cudaEventDestroy(start[j]);
    cudaEventDestroy(stop[j]); 
  }


  float func_netIsend_time = (float)(netIsend_time - nccl_func_start_time) / CLOCKS_PER_SEC * 1000.0f; 

  float netIrecv_func_time = (float)(nccl_func_end_time - netIrecv_time) / CLOCKS_PER_SEC * 1000.0f; 

  float nccl_func_time = (float)(nccl_func_end_time - nccl_func_start_time) / CLOCKS_PER_SEC * 1000.0f;  


  #endif

  ////////////////////////////// PROFILE_LYD_P2P_DEVICE_SYNC: END //////////////////////////////

  ////////////////////////////// PROFILE_LYD_P2P_HOST PRINT TIME : START //////////////////////////////

  #if PROFILE_LYD_P2P_HOST_SYNC == 1 || PROFILE_LYD_P2P_HOST_GROUP == 1
  float gauge_time;
  cudaEventElapsedTime(&gauge_time, p2p_time_stamp[0], p2p_time_stamp[N_ITERS]);
  if (myRank == 0) { 
    printf("heo(%s)_mode(%s)_nchannels(%s)_chunk size(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f ms\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_chunk_size_var, env_gauge_size_var, N_ITERS, loggp_gauge_d, env_gauge_iteration_var, gauge_time);
  }

  // Clean up
  for (int i = 0; i < (N_ITERS+1); ++i) {
    cudaEventDestroy(p2p_time_stamp[i]); // Initialize each event individually
  }
  #endif

  ////////////////////////////// PROFILE_LYD_P2P_HOST PRINT TIME : END //////////////////////////////

  //completing NCCL operation by synchronizing on the CUDA stream
  // CUDACHECK(cudaStreamSynchronize(s));

  for (int j = 0; j < N_STREAMS; ++j) {
      CUDACHECK(cudaStreamSynchronize(streams[j]));
  }

  if (myRank < 2) {
    sprintf(filename, "%s/nccl_pping_%s_chunk-%s_r-%d_e-%s.out", env_gauge_output_dir_var, env_gauge_heo_var, env_gauge_chunk_size_var, myRank, env_experiment_id_var);
    freopen(filename, "a", stdout);
  } else {
    freopen("/dev/null", "w", stdout);
  }

  // After the kernel execution, copy the messages back to the host
  LogMessage_lyd* h_messages = new LogMessage_lyd;
  cudaMemcpy(h_messages, d_messages, sizeof(LogMessage_lyd), cudaMemcpyDeviceToHost);

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
  // if (myRank == 0){
  //   printf("DEVICE | sendrecv.h | full recv - send | time: %f us\n", h_messages->timeValue2[0][3]-h_messages->timeValue2[0][0]);
  //   for (size_t i = 0; i < maxMessages; ++i) {
  //     for (size_t j = 0; j < MAXLOGLYD; j++){
  //       if (j>0) printf("DEVICE | sendrecv.h | runsend%d - runsend0 | warp %d | iteration %d | time: %f us\n", j, i, j, h_messages->timeValue[i][j] - h_messages->timeValue[i][0]);
  //       printf("DEVICE | sendrecv.h | runrecv - runsend | warp %d | iteration %d | time: %f us\n", i, j, h_messages->timeValue1[i][j] - h_messages->timeValue[i][j]);
  //     }
  //   }
  // } else {
  //   printf("DEVICE | sendrecv.h | full send - recv | time: %f us\n", h_messages->timeValue2[0][0]-h_messages->timeValue2[0][3]);
  //   for (size_t i = 0; i < maxMessages; ++i) {
  //     for (size_t j = 0; j < MAXLOGLYD; j++){
  //       printf("DEVICE | sendrecv.h | runsend - runrecv | warp %d | iteration %d | time: %f us\n", i, j, h_messages->timeValue[i][j] - h_messages->timeValue1[i][j]);
  //     }
  //   }
  // }

  double gauge_time;

  if (myRank == 0) {
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_d(%d)_iteration(%s)_nccl pping elapsed time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, elapsed_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_d(%d)_iteration(%s)_nccl pping elapsed time by clock: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, nccl_func_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_d(%d)_iteration(%s)_nccl func to netIsend time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, func_netIsend_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_d(%d)_iteration(%s)_nccl ncclIrecv to func time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, netIrecv_func_time);
    gauge_time = static_cast<double>(h_messages->timeEndValue[1] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
    printf("device_time_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_d(%d)_iteration(%s): %f us\n", env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, GAUGE_D, env_gauge_iteration_var, gauge_time);
    for (size_t i = 0; i < N_CHUNKS; ++i) { 
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
      printf("heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, GAUGE_D, env_gauge_iteration_var, gauge_time);
    }
  } else {
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_d(%d)_iteration(%s)_nccl pping elapsed time by clock: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, nccl_func_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_d(%d)_iteration(%s)_nccl func to netIsend time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, func_netIsend_time);
    printf("message size(%s)_nchannels(%s)_nthreads(%s)_n(%d)_d(%d)_iteration(%s)_nccl ncclIrecv to func time: %f ms\n", env_gauge_size_var, env_gauge_nchannels_var, env_gauge_nthreads_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, netIrecv_func_time);
    for (size_t i = 0; i < N_CHUNKS; ++i) { 
      gauge_time = static_cast<double>(h_messages->timeValue[0][i] - h_messages->timeValue[1][0]) / GAUGE_GPU_FREQUENCY;
      printf("heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f us\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, GAUGE_D, env_gauge_iteration_var, gauge_time);
    }
  } 

  // print the gap between chunks
  if (myRank == 0) { 
    for (size_t i = 0; i < N_CHUNKS; ++i) {
      gauge_time = static_cast<double>(h_messages->timeValue[0][i] - h_messages->timeValue[0][0]) / GAUGE_GPU_FREQUENCY;
      printf("chunk gap | chunk%d -> chunk0 | heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f us\n", i, env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, GAUGE_D, env_gauge_iteration_var, gauge_time);
    }
  } else {
    for (size_t i = 0; i < N_CHUNKS; ++i) {
      gauge_time = static_cast<double>(h_messages->timeValue[1][i] - h_messages->timeValue[1][0]) / GAUGE_GPU_FREQUENCY;
      printf("chunk gap | chunk%d -> chunk0 | heo(%s)_mode(%s)_nchannels(%s)_nthreads(%s)_chunk steps(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f us\n", i, env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_nthreads_var, env_gauge_chunk_size_var, env_gauge_size_var, i, GAUGE_D, env_gauge_iteration_var, gauge_time);
    }
  }
  #endif

  // Free the device memory of the gauge test
  cudaFree(d_messages);
  delete[] h_messages;


  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));


  // //finalizing NCCL
  // ncclCommDestroy(comm);

  for (int i = 0; i < N_STREAMS; ++i) {
    ncclCommDestroy(comm[i]);
  }

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}