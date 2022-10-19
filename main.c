/* matrix multiplication with strassen alg */
/* matrix is stored with row-major order. */
/* Written by Junhong Zhang */

// C = A*B
// (m*n) = (m*p) * (p*n);
// where we assume m = n = p;

#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "gemm.h"F

int main(int argc, char *argv[])
{
    int m, n, p;
    if (argc >= 2) {
        m = atoi(argv[1]);
    }
    else {
        m = 1024;
    }
    
    // init square matrix
    p = n = m;

    // leading dimension
    int lda = m; 
    int ldb = n;
    int ldc = m;

    // memory allocatino
    fdata *a   = ALLOC_ARR(m*lda);
    fdata *b   = ALLOC_ARR(n*ldb);
    fdata *c   = ALLOC_ARR(m*ldc);
    fdata *ref = ALLOC_ARR(m*ldc);
    struct timeval start, end;

    rand_matrix(a, m, p, lda);
    rand_matrix(b, p, n, ldb);

    // naive gemm
    gettimeofday(&start, NULL);
    naive_gemm(m, n, p, a, lda, b, ldb, ref, ldc);
    gettimeofday(&end, NULL);
    printf("Trivial: %.6f sec\n", ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6);

    // divide and conquer method
    gettimeofday(&start, NULL);
    naive_dc_gemm(m, n, p, a, lda, b, ldb, c, ldc);
    gettimeofday(&end, NULL);
    printf("DivConq: %.6f sec\n", ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6);

    // strassen
    gettimeofday(&start, NULL);
    strassen_gemm(m, n, p, a, lda, b, ldb, c, ldc);
    gettimeofday(&end, NULL);
    printf("Strassen: %.6f sec\n", ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6);

    // check the result
    double err = matrix_diff(c, ref, m, n, lda);
    printf("max elem-wise error = %g\n", err);
    if (err > 1e-10) {
        puts("ERROR");
    }
    else {
        puts("CORRECT");
    }

    free(a);
    free(b);
    free(c);
    free(ref);

    return EXIT_SUCCESS;
}