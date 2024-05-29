#ifndef MATRIX_H
#define MATRIX_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

typedef struct
{
    double * m;
    unsigned columns;
    unsigned rows;
}  matrix_t;

matrix_t * alloc_matrix(unsigned rows, unsigned columns);

void destroy_matrix(matrix_t *m);

void print_matrix(matrix_t *m, bool is_short);

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res);

__global__ 
void matrixMultiplyKernel(double *A, double *B, double *res, int m, int n, int k);

__global__
void matrixMultiplyKernelManaged(double *A, double *B, double *res, int m, int n, int k);

void matrixDotCUDAManaged(int m, int n, int k);
void matrixDotCUDAManaged2(double *h_A, double *h_B, double *h_res, int m, int n, int k);
//void matrixDotCUDAManaged(double *h_A, double *h_B, double *h_res, int m, int n, int k);

void matrixDotCUDAManaged3(matrix_t *A, matrix_t *B, matrix_t *C);

void matrixDotCUDA(double *h_A, double *h_B, double *h_res, int m, int n, int k);

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res);

void matrix_transpose(matrix_t *m1, matrix_t *res);

void matrix_scalar(matrix_t *m1, double s, matrix_t *res);

void matrix_memcpy(matrix_t *dest, const matrix_t *src);

#endif