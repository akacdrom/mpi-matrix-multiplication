#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <time.h>
#define N 2000

// Allocate arrays in Heap memory.
auto firstMatrix = new int[N][N];
auto secondMatrix = new int[N][N];

decltype(auto) serialAlgorithm(int firstMatrix[][N], int secondMatrix[][N])
{
    int sum = 0;

    // Array of pointers to arrays
    auto newSequentialMatrix = new int [N][N];


    printf("Sequential Matrix multiplication started...\n");
    clock_t t;
    t = clock();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                sum = sum + firstMatrix[i][k] * secondMatrix[k][j];
            }
            newSequentialMatrix[i][j] = sum;
            sum = 0;
        }
    }
    t = clock() - t;
    printf("Sequential Matrix multiplication is finished! ===> %f ms. \n", ((float)t) / CLOCKS_PER_SEC);

    // Print sequential matrix
    // for (int i = 0; i < N; i++)
    // {
    //     for (int k = 0; k < N; k++)
    //     {
    //         printf("%d ", newSequentialMatrix[i][k]);
    //     }
    //     printf("\n");
    // }
    return newSequentialMatrix;
}

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    int my_rank, size, root = 0, sum = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = N / size;

    // Allocate arrays in Heap memory.
    auto recvFirstMatrix = new int[N][N];
    auto newParallelMatrix = new int[N][N];
    auto newParallelMatrixGather = new int[N][N];
    double minTime, maxTime, avgTime;

    if (my_rank == 0)
    {

        // Create First Matrix
        printf("First Matrix is being created...\n");
        firstMatrix[N][N];
        srand(time(NULL));
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < N; k++)
            {
                firstMatrix[i][k] = rand() % 9 + 1;
            }
        }

        // Print First Matrix
        // for (int i = 0; i < N; i++)
        // {
        //     for (int k = 0; k < N; k++)
        //     {
        //         printf("%d ", firstMatrix[i][k]);
        //     }
        //     printf("\n");
        // }
        printf("First Matrix is created!\n");

        // Create Second Matrix
        printf("Second Matrix is being created...\n");
        secondMatrix[N][N];
        srand(time(NULL)+1);
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < N; k++)
            {
                secondMatrix[i][k] = rand() % 9 + 1;
            }
        }

        // Print Second Matrix
        // for (int i = 0; i < N; i++)
        // {
        //     for (int k = 0; k < N; k++)
        //     {
        //         printf("%d ", secondMatrix[i][k]);
        //     }
        //     printf("\n");
        // }
        printf("Second Matrix is created!\n");
        printf("Parallel Matrix multiplication started...\n");

        // Sleep a second to get different random number by time.
    }

    MPI_Scatter(firstMatrix, N * N / size, MPI_INT, recvFirstMatrix, N * N / size, MPI_INT, root, MPI_COMM_WORLD);

    // broadcast second matrix to all processes
    MPI_Bcast(secondMatrix, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    printf("    --> Current Rank Is: %d \n", my_rank);
    double startTime = MPI_Wtime();

    for (int i = 0; i < chunk; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                sum += recvFirstMatrix[i][k] * secondMatrix[k][j];
            }
            newParallelMatrix[i][j] = sum;
            sum = 0;
        }
    }
    double elapsedTime = MPI_Wtime() - startTime;

    MPI_Gather(newParallelMatrix, N * N / size, MPI_INT, newParallelMatrixGather, N * N / size, MPI_INT, root, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Time measurement
    MPI_Reduce(&elapsedTime, &minTime, 1, MPI_DOUBLE, MPI_MIN, root, MPI_COMM_WORLD);
    MPI_Reduce(&elapsedTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
    MPI_Reduce(&elapsedTime, &avgTime, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("Parallel Matrix multiplication is finished! ===> Min: %f | Max: %f | Avg: %f\n", minTime, maxTime, avgTime / 4);

        // printf("Parallel Matrix multiplication Results ---> \n");
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         printf("%d ", newParallelMatrixGather[i][j]);
        //     }
        //     printf("\n");
        // }

        // serialAlgorithm(firstMatrix, secondMatrix);
    }

    MPI_Finalize();
}