#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <time.h>
#define N 10

int sum = 0;

// decltype keyword used to return auto data typed array.
decltype(auto) SerialAlgorithm(int first_matrix[][N], int second_matrix[][N])
{
    auto new_sequential_matrix = new int[N][N];

    printf("Sequential Matrix multiplication started...\n");
    clock_t t;
    t = clock();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                sum = sum + first_matrix[i][k] * second_matrix[k][j];
            }
            new_sequential_matrix[i][j] = sum;
            sum = 0;
        }
    }
    t = clock() - t;
    printf("Sequential Matrix multiplication is finished! ===> %f ms. \n", ((float)t) / CLOCKS_PER_SEC);

    // Print sequential matrix
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            printf("%d ", new_sequential_matrix[i][k]);
        }
        printf("\n");
    }
    return new_sequential_matrix;
}

decltype(auto) ParallelAlgorithm(int first_matrix[][N], int second_matrix[][N],
                                 int my_rank)
{
    int size, root = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // matrix rows equally divided to threads.
    int chunk = N / size;

    // allocate arrays in Heap memory.
    auto recv_first_matrix = new int[N][N];
    auto new_parallel_matrix = new int[N][N];
    auto new_parallel_matrix_gather = new int[N][N];

    // double data type to get ms.
    double min_time, max_time, avg_time;

    // get first matrix to all process as whole.
    MPI_Scatter(first_matrix, N * N / size, MPI_INT, recv_first_matrix, N * N / size, MPI_INT, root, MPI_COMM_WORLD);

    // broadcast second matrix to all processes
    MPI_Bcast(second_matrix, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    printf("    --> Rank %d is STARTED to job.\n", my_rank);
    double start_time = MPI_Wtime();

    for (int i = 0; i < chunk; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                sum += recv_first_matrix[i][k] * second_matrix[k][j];
            }
            new_parallel_matrix[i][j] = sum;
            printf("element: %d rank-> %d | ", sum, my_rank);
            sum = 0;
        }
        printf("\n");
    }
    double elapsed_time = MPI_Wtime() - start_time;
    printf("    --> Rank %d is FINISHED to job. It took it %f ms.\n", my_rank, elapsed_time);

    MPI_Gather(new_parallel_matrix, N * N / size, MPI_INT, new_parallel_matrix_gather, N * N / size, MPI_INT, root, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Time measurement
    MPI_Reduce(&elapsed_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, root, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("Parallel Matrix multiplication is finished! ===> Min: %f | Max: %f | Avg: %f\n", min_time, max_time, avg_time / size);

        // printf("Parallel Matrix multiplication Results ---> \n");
        // for (int i = 0; i < N; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //     {
        //         printf("%d ", new_parallel_matrix_gather[i][j]);
        //     }
        //     printf("\n");
        // }
    }
    return new_parallel_matrix_gather;
}

int main(int argc, char **argv)
{
    // Allocate constant arrays in Heap memory.
    auto first_matrix = new int[N][N];
    auto second_matrix = new int[N][N];

    MPI_Init(NULL, NULL);

    int my_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // If it is master create random arrays
    // I might implenet "MPI_Iprobe()" for less aggressive polling while creation of arrays.
    if (my_rank == 0)
    {
        // Create First Matrix
        printf("First Matrix is being created...\n");
        first_matrix[N][N];
        srand(time(NULL));
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < N; k++)
            {
                first_matrix[i][k] = rand() % 9 + 1;
            }
        }

        // Print First Matrix
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < N; k++)
            {
                printf("%d ", first_matrix[i][k]);
            }
            printf("\n");
        }
        printf("First Matrix is created!\n");

        // Create Second Matrix
        printf("Second Matrix is being created...\n");
        second_matrix[N][N];
        srand(time(NULL) + 1);
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < N; k++)
            {
                second_matrix[i][k] = rand() % 9 + 1;
            }
        }

        // Print Second Matrix
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < N; k++)
            {
                printf("%d ", second_matrix[i][k]);
            }
            printf("\n");
        }
        printf("Second Matrix is created!\n");
        printf("Parallel Matrix multiplication started...\n");
    }

    // Wait for master process to finish create matrises
    MPI_Barrier(MPI_COMM_WORLD);

    // Both matrises and other arguments passed to parallel algorithm function
    ParallelAlgorithm(first_matrix, second_matrix, my_rank);

    // Wait for all processes to finish parallel algorithm.
    MPI_Barrier(MPI_COMM_WORLD);

    // I called serial algorithm before MPI_Finalize for not make unnecessary polling to other threads on CPU
    if (my_rank == 0)
    {
        // both matrises passed to serial algorithm function
        SerialAlgorithm(first_matrix, second_matrix);
    }
    MPI_Finalize();
}