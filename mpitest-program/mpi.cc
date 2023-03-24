#include <mpi.h>
#include <stdio.h>

int main (int argc, char* argv[]) {
    int rank, size;
    
    // Print debug
    printf("[DEBUG] Main function started...\n");

    // Initialize MPI
    MPI_Init (&argc, &argv);
    printf("[DEBUG] MPI_Init completed.\n");
    
    // Get the current process ID
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    printf("[DEBUG] MPI_Comm_rank completed.\n");
    
    // Get the number of processors in the world
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    printf("[DEBUG] MPI_Comm_size completed.\n");
    
    // Print debug statement
    printf("Hello world from process %d of %d\n", rank, size);
    
    // Test communication
    if (rank == 0) {
        // Create a test string
        char testString[12] = "Test rank 0";
        MPI_Send(&testString, 12, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Status status;
        // Create a test string
        char testString[12];
        MPI_Recv(&testString, 12, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        printf("Rank %d got: ", rank);  
        printf(testString);
        printf("\n");
    } else {
        printf("Rank %d won't do anything.", rank);
    }
    
    // Finalize MPI
    MPI_Finalize();
    printf("[DEBUG] MPI_Finalize completed.\n");
    
    // Exit program
    return 0;
}