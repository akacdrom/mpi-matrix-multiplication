#include <iostream>
// I'm not sure this works on windows should check it!
#include <unistd.h>
#include <time.h> /* clock_t, clock, CLOCKS_PER_SEC */
#include <mpi.h>
#include <stdio.h>

using namespace std;

const static int MATRIX_ROW_COLUMN = 1000;

static int **serialAlgorithm(int firstMatrix[][MATRIX_ROW_COLUMN], int secondMatrix[][MATRIX_ROW_COLUMN])
{
    // I have standart algorithm for create the serial algorithm.
    int s = 0;
    int **newSequentialMatrix = new int *[MATRIX_ROW_COLUMN];
    for (int i = 0; i < MATRIX_ROW_COLUMN; i++)
    {
        newSequentialMatrix[i] = new int[MATRIX_ROW_COLUMN];
    }

    cout << "Sequential Matrix multiplication started...\n";
    clock_t t;
    t = clock();
    for (int i = 0; i < MATRIX_ROW_COLUMN; i++)
    {
        for (int j = 0; j < MATRIX_ROW_COLUMN; j++)
        {
            for (int k = 0; k < MATRIX_ROW_COLUMN; k++)
            {
                s = s + firstMatrix[i][k] * secondMatrix[k][j];
            }
            newSequentialMatrix[i][j] = s;
            s = 0;
        }
    }
    t = clock() - t;
    cout << "Sequential Matrix multiplication is finished! ===> " << ((float)t) / CLOCKS_PER_SEC << " ms.\n";
    return newSequentialMatrix;
}

static void parallelAlgorithm(int firstMatrix[][MATRIX_ROW_COLUMN], int secondMatrix[][MATRIX_ROW_COLUMN])
{
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
}

int main(int argc, char **argv)
{
    // I needed change my stack size for work with large arrays.
    // "ulimit -s 1024" is works for me.
    cout << "First Matrix is being created...\n";
    static int firstMatrix[MATRIX_ROW_COLUMN][MATRIX_ROW_COLUMN];
    srand(time(NULL));
    for (int i = 0; i < MATRIX_ROW_COLUMN; i++)
    {
        for (int k = 0; k < MATRIX_ROW_COLUMN; k++)
        {
            firstMatrix[i][k] = rand() % 9 + 1;
        }
    }
    cout << "First Matrix is created!\n";

    sleep(1);

    cout << "Second Matrix is being created...\n";
    static int secondMatrix[MATRIX_ROW_COLUMN][MATRIX_ROW_COLUMN];
    srand(time(NULL));
    for (int i = 0; i < MATRIX_ROW_COLUMN; i++)
    {
        for (int k = 0; k < MATRIX_ROW_COLUMN; k++)
        {
            secondMatrix[i][k] = rand() % 9 + 1;
        }
    }
    cout << "Second Matrix is created!\n";
    parallelAlgorithm(firstMatrix, secondMatrix);
    serialAlgorithm(firstMatrix, secondMatrix);
}
