#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

// Helper function to initialize the data buffer
void initData(float* buffer, int size, float value) {
    for (int i = 0; i < size; i++) {
        buffer[i] = value;
    }
}

int main(int argc, char* argv[]) {
    int myRank, nRanks, size = 1024;  // Size of the message in number of floats
    float *sendbuff, *recvbuff;

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

    const char* env_gauge_size_var = getenv("GAUGE_MESSAGE_SIZE");
    if (env_gauge_size_var != nullptr) {
        size = atoi(env_gauge_size_var) * 1024 / 4;  // Convert from kilobytes to number of floats, assuming the environment variable is in kilobytes
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    char filename[256];

    if (myRank < 2) {
      sprintf(filename, "%s/nccl_pping_%s_chunk%s_itr0-5-mpi-r%d.out", env_gauge_output_dir_var, env_gauge_heo_var, env_gauge_chunk_size_var, myRank);
      freopen(filename, "a", stdout);
    } else {
      freopen("/dev/null", "w", stdout);
    }

    // Allocate memory for send and receive buffers
    sendbuff = (float*)malloc(size * sizeof(float));
    recvbuff = (float*)malloc(size * sizeof(float));

    initData(sendbuff, size, myRank * 1.0f);  // Initialize send buffer with unique values per rank

    int recvPeer = (myRank - 1 + nRanks) % nRanks;
    int sendPeer = (myRank + 1) % nRanks;

    MPI_Status status;

    double MpiTime[N_ITERS+1];

    // for (int i = 0; i < N_ITERS; i++) {
    //   MpiTime[i] = MPI_Wtime();
    //   if (i != 0) sleep(GAUGE_D);
    //   if (myRank == 0) {
    //       MPI_Send(sendbuff, size, MPI_FLOAT, sendPeer, 0, MPI_COMM_WORLD);
    //   } else {
    //       MPI_Recv(recvbuff, size, MPI_FLOAT, recvPeer, 0, MPI_COMM_WORLD, &status);
    //   }
    // }

    // if (myRank == 1) {
    //     MPI_Send(sendbuff, size, MPI_FLOAT, sendPeer, 0, MPI_COMM_WORLD);
    // } else {
    //     MPI_Recv(recvbuff, size, MPI_FLOAT, recvPeer, 0, MPI_COMM_WORLD, &status);
    // }

    if (myRank == 0) {
      usleep(10);
      MpiTime[0] = MPI_Wtime();
      for (int i = 0; i < N_ITERS; i++) {
        // MpiTime[i] = MPI_Wtime();
        if (i != 0) sleep(GAUGE_D);
        MPI_Send(sendbuff, size, MPI_FLOAT, sendPeer, 0, MPI_COMM_WORLD);
      }
      MPI_Recv(recvbuff, size, MPI_FLOAT, recvPeer, 0, MPI_COMM_WORLD, &status);
      MpiTime[N_ITERS] = MPI_Wtime();
    } else {
      MpiTime[0] = MPI_Wtime();
      for (int i = 0; i < N_ITERS; i++) {
        // MpiTime[i] = MPI_Wtime();
        MPI_Recv(recvbuff, size, MPI_FLOAT, recvPeer, 0, MPI_COMM_WORLD, &status);
      }
      MPI_Send(sendbuff, size, MPI_FLOAT, sendPeer, 0, MPI_COMM_WORLD);
      MpiTime[N_ITERS] = MPI_Wtime();
    }

    free(sendbuff);
    free(recvbuff);

    MPI_Finalize();

    printf("[MPI Rank %d] Success \n", myRank);

    // for (int i = 0; i < N_ITERS; i++) {
    //   printf("[MPI Rank %d] Gap Time: %f\n", myRank, MpiTime[i+1] - MpiTime[i]);
    // }

    float gauge_time;
    gauge_time = (MpiTime[N_ITERS] - MpiTime[0]) * 1000;

    if (myRank == 0) { 
      printf("heo(%s)_mode(%s)_nchannels(%s)_chunk size(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f ms\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_chunk_size_var, env_gauge_size_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, gauge_time);
    } else {
      printf("heo(%s)_mode(%s)_nchannels(%s)_chunk size(%s)_message size(%s)_n(%d)_d(%d)_iteration(%s): %f ms\n", env_gauge_heo_var, env_gauge_mode_var, env_gauge_nchannels_var, env_gauge_chunk_size_var, env_gauge_size_var, N_ITERS, GAUGE_D, env_gauge_iteration_var, gauge_time);
    }



    return 0;
}
