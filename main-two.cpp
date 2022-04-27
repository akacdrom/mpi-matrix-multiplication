#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <time.h>
#include <array>

using namespace std;

const int matrix_row_column = 500;

// To return matrises
using Row = std::array<int, matrix_row_column>;
using Matrix = std::array<Row, matrix_row_column>;
Matrix parallel_array;
Matrix serial_array;

static int array_first[matrix_row_column][matrix_row_column];
static int array_second[matrix_row_column][matrix_row_column];
static int local_matrix[matrix_row_column][matrix_row_column];
static int parallel_array_temp[matrix_row_column][matrix_row_column];

Matrix *Parallel(int array_first[][matrix_row_column], int array_second[][matrix_row_column],
                 int rank, int chunk, int size)
{
    int s = 0;

    MPI_Scatter(array_first, matrix_row_column * matrix_row_column / size, MPI_INT,
                local_matrix, matrix_row_column * matrix_row_column / size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(array_second, matrix_row_column * matrix_row_column, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++)
    {
        for (int j = 0; j < matrix_row_column; j++)
        {
            for (int k = 0; k < matrix_row_column; k++)
            {
                s = s + local_matrix[k][i] * array_second[k][j];
            }
            parallel_array_temp[i][j] = s;
            s = 0;
        }
    }

    MPI_Gather(parallel_array_temp, matrix_row_column * matrix_row_column / size, MPI_INT,
               &parallel_array, matrix_row_column * matrix_row_column / size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        return &parallel_array;
    }
    return 0;
}

Matrix *Sequential(int array_first[][matrix_row_column], int array_second[][matrix_row_column])
{
    int s = 0;
    for (int i = 0; i < matrix_row_column; i++)
    {
        for (int j = 0; j < matrix_row_column; j++)
        {
            for (int k = 0; k < matrix_row_column; k++)
            {
                s = s + array_first[k][i] * array_second[k][j];
            }
            serial_array[i][j] = s;
            s = 0;
        }
    }

    return &serial_array;
}

// int (*CreateFirstArray())[matrix_row_column]
void CreateSecondMatrix()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    srand(time(0));
    for (int i = 0; i < matrix_row_column; i++)
    {
        for (int k = 0; k < matrix_row_column; k++)
        {
            array_first[i][k] = rand() % 100 + 10;
        }
    }
}

// int (*CreateSecondArray())[matrix_row_column]
void CreateFirstMatrix()
{
    srand(time(0));
    for (int i = 0; i < matrix_row_column; i++)
    {
        for (int k = 0; k < matrix_row_column; k++)
        {
            array_second[i][k] = rand() % 100 + 10;
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = matrix_row_column / size;

    if (rank == 0)
    {
        CreateFirstMatrix();
        CreateSecondMatrix();
        double sequential_start = MPI_Wtime();
        Sequential(array_first, array_second);
        double stop_sequential = MPI_Wtime();
        std::cout << "sequential: " << stop_sequential - sequential_start << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double parallel_start = MPI_Wtime();

    Parallel(array_first, array_second, rank, chunk, size);

    if (rank == 0)
    {
        double parallel_stop = MPI_Wtime();
        std::cout << "parallel: " << parallel_stop - parallel_start;
    }

    MPI_Finalize();
}