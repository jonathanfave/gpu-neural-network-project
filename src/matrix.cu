#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

matrix_t * alloc_matrix2(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix2(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    free(m->m);
    free(m);
}

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res;
    cudaMallocManaged( (void **) &res, sizeof(matrix_t));
    cudaMallocManaged( (void **) &(res->m), columns * rows *sizeof(double));
    cudaMemset(res->m, 0, res->columns * res->rows * sizeof(double));
    
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    cudaFree(m->m);
    cudaFree(m);
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}

// Kernel for matrix multiplication
__global__ 
void matrixMultiplyKernel(double *A, double *B, double *res, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }   
        res[row * k + col] = sum;
    }
}

// Kernel for matrix multiplication using shared memory
__global__
void matrixMultiplyKernelManaged(double *A, double *B, double *res, int m, int n, int k) {
    __shared__ double s_A[16][16];
    __shared__ double s_B[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    for (int tile = 0; tile < (n + 15) / 16; ++tile) {
        if (row < m && tile * 16 + threadIdx.x < n) {
            s_A[threadIdx.y][threadIdx.x] = A[row * n + tile * 16 + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < k && tile * 16 + threadIdx.y < n) {
            s_B[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * k + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < 16; ++i) {
            sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < k) {
        res[row * k + col] = sum;
    }
}

// Function to perform matrix multiplication using CUDA with unified memory
void matrixDotCUDAManaged3(matrix_t *A, matrix_t *B, matrix_t *C) {
    int m = A->rows;
    int n = A->columns;
    int k = B->columns;

    // Définition des dimensions du block et de la grille
    dim3 dimBlock(16, 16);
    dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    // Lancement du kernel
    matrixMultiplyKernelManaged<<<dimGrid, dimBlock>>>(A->m, B->m, C->m, m, n, k);
    cudaDeviceSynchronize();
}

// Function to perform matrix multiplication using CUDA with unified memory
void matrixDotCUDAManaged2(double *h_A, double *h_B, double *h_res, int m, int n, int k) {
    // Allocate unified memory
    double *um_A, *um_B, *um_res;
    int size_A = m * n * sizeof(double);
    int size_B = n * k * sizeof(double);
    int size_res = m * k * sizeof(double);
    cudaMallocManaged(&um_A, size_A);
    cudaMallocManaged(&um_B, size_B);
    cudaMallocManaged(&um_res, size_res);

    // Copy data to unified memory
    //memcpy(um_A, h_A, size_A);
    //memcpy(um_B, h_B, size_B);

    // Define block and grid sizes
    dim3 dimBlock(32, 32);
    dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    matrixMultiplyKernelManaged<<<dimGrid, dimBlock>>>(um_A, um_B, um_res, m, n, k);
    cudaDeviceSynchronize(); // Wait for kernel to finish execution

    // Copy result back to host memory
    //memcpy(h_res, um_res, size_res);

    // Free unified memory
    cudaFree(um_A);
    cudaFree(um_B);
    cudaFree(um_res);
}

// Function to perform matrix multiplication using CUDA with unified memory
void matrixDotCUDAManaged(int m, int n, int k) {
    // Allocate unified memory
    double *um_A, *um_B, *um_res;
    int size_A = m * n * sizeof     ( double );
    int size_B = n * k * sizeof     ( double );
    int size_res = m * k * sizeof   ( double );
    cudaMallocManaged(&um_A, size_A);
    cudaMallocManaged(&um_B, size_B);
    cudaMallocManaged(&um_res, size_res);

    // Define block and grid sizes
    dim3 dimBlock(16, 16);
    dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(um_A, um_B, um_res, m, n, k);
    cudaDeviceSynchronize(); // Wait for kernel to finish execution

    // Free unified memory
    cudaFree(um_A);
    cudaFree(um_B);
    cudaFree(um_res);
}


// Function to perform matrix multiplication using CUDA
void matrixDotCUDA(double *h_A, double *h_B, double *h_res, int m, int n, int k) {
    int size_A = m * n * sizeof     ( double );
    int size_B = n * k * sizeof     ( double );
    int size_res = m * k * sizeof   ( double );

    // cudaMalloc pour les host ? 

    // Allocate memory on the GPU
    double *d_A, *d_B, *d_res;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_res, size_res);

    // Transfer data from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 dimBlock(16, 16);
    dim3 dimGrid((k + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    // Launch kernel
    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_res, m, n, k);

    // Transfer result from device to host
    cudaMemcpy(h_res, d_res, size_res, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_res);
    // Free memoire host ??
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {   
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}