#include "julius/julius.hpp"
#include "julius/utils.hpp"

#include <iostream>
#include <cmath>

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

    int warm_loop = 10;
    for (int i = 0; i < warm_loop; ++i)
    {
        // julius cblas
        cblas_sgemv(order, trans, M, N, alpha, A, lda, x, incx, beta, y_cblas, incy);

        // naive
        // juliusblas::cblas_sgemv_naive(order, trans, M, N, alpha, A, lda, x, incx, beta, y_naive, incy);
    }

    int loop_count = 100;

    juliusblas::timer t;
    double cost_time = 0;
    double max_time = INT32_MIN;
    double min_time = INT32_MAX;

    for (int i = 0; i < loop_count; ++i)
    {
        t.start();

        // julius cblas
        cblas_sgemv(order, trans, M, N, alpha, A, lda, x, incx, beta, y_cblas, incy);

        // naive
        // juliusblas::cblas_sgemv_naive(order, trans, M, N, alpha, A, lda, x, incx, beta, y_naive, incy);

        t.stop();

        double time = t.get_elapsed_milli_seconds();
        max_time = std::max(max_time, time);
        min_time = std::min(min_time, time);
        cost_time += time;
    }

    std::cout << "MAX\tMIN\tAVG" << std::endl;
    std::cout << max_time << "\t" << min_time << "\t" << cost_time / loop_count << std::endl;

    delete[] A;
    delete[] x;
    delete[] y_cblas;
    delete[] y_naive;

    return 0;
}