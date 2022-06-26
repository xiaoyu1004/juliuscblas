#include "julius/julius.hpp"
#include "julius/utils.hpp"

#include <iostream>

int main()
{
    const enum CBLAS_LAYOUT order = CblasRowMajor;
    const enum CBLAS_TRANSPOSE trans = CblasNoTrans;

    const float alpha = 1.f;

    int M = 1 << 14;
    int N = 1 << 14;

    const int lda = N;
    const int incx = 1;
    const float beta = 0.f;
    const int incy = 1;

    float *A = new float[M * N];
    float *x = new float[N];

    float *y_cblas = new float[M];
    float *y_naive = new float[M];

    juliusblas::InitialMatrix<float>(M, N, A);
    juliusblas::InitialMatrix<float>(N, 1, x);

    // print input
    // juliusblas::PrintMatrix(order, M, N, A);
    // juliusblas::PrintMatrix(order, N, 1, x);

    // julius cblas
    cblas_sgemv(order, trans, M, N, alpha, A, lda, x, incx, beta, y_cblas, incy);

    // naive
    juliusblas::cblas_sgemv_naive(order, trans, M, N, alpha, A, lda, x, incx, beta, y_naive, incy);

    // print
    // juliusblas::PrintMatrix(order, 1, M, Y);

    // compare
    juliusblas::CompareResult(M, y_cblas, y_naive);

    delete[] A;
    delete[] x;
    delete[] y_cblas;
    delete[] y_naive;

    return 0;
}