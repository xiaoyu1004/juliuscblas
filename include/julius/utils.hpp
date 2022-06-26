#ifndef UTILS_HPP
#define UTILS_HPP

#include "julius/julius.hpp"

#include <iostream>
#include <chrono>

namespace juliusblas
{
    template <typename T>
    void PrintMatrix(const enum CBLAS_LAYOUT order, int m, int n, const T *ptr)
    {
        if (order == CblasRowMajor)
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    std::cout << ptr[i * n + j] << "\t";
                }
                std::cout << std::endl;
            }
        }
        else if (order == CblasColMajor)
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    std::cout << ptr[j * m + i] << "\t";
                }
                std::cout << std::endl;
            }
        }
        else
        {
            LOGE(" Errors in Julius print matrix.\n");
        }
    }

    template <typename T>
    void InitialMatrix(int m, int n, T *ptr)
    {
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                ptr[i * n + j] = 1.f;
            }
        }
    }

    void cblas_sgemv_naive(const enum CBLAS_LAYOUT order, const enum CBLAS_TRANSPOSE trans, const int M, const int N,
                           const float alpha, const float *A, const int lda, const float *x, const int incx, const float beta, float *y, const int incy)
    {
        if (order == CblasRowMajor)
        {
            if (trans == CblasNoTrans)
            {
                for (int i = 0; i < M; ++i)
                {
                    y[i * incy] = beta * y[i * incy];
                    for (int j = 0; j < N; ++j)
                    {
                        y[i * incy] += alpha * A[i * N + j] * x[j * incx];
                    }
                }
            }
        }
    }

    template <typename T>
    void CompareResult(const int len, const T *r1, const T *r2)
    {
        for (int i = 0; i < len; ++i)
        {
            if (std::abs(r1[i] - r2[i]) > 1e-2)
            {
                LOGE("Error: compare result is incorrect!\n");
            }
        }

        LOGI("Success: compare result is correct!\n");
    }

    class timer
	{
	public:
		timer() : start_(), end_()
		{
		}

		void start()
		{
			start_ = std::chrono::system_clock::now();
		}

		void stop()
		{
			end_ = std::chrono::system_clock::now();
		}

		double get_elapsed_seconds() const
		{
			return (double)std::chrono::duration_cast<std::chrono::seconds>(end_ - start_).count();
		}

		double get_elapsed_milli_seconds() const
		{
			return (double)std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
		}

		double get_elapsed_micro_seconds() const
		{
			return (double)std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();
		}

		double get_elapsed_nano_seconds() const
		{
			return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_).count();
		}

	private:
		std::chrono::time_point<std::chrono::system_clock> start_;
		std::chrono::time_point<std::chrono::system_clock> end_;
	};
} // juliusblas

#endif