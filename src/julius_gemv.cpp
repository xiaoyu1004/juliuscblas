#include "julius_gemv.hpp"
#include "utils/simd_types.hpp"

namespace juliusblas
{
    void packTransedA(const int M, const int N, const float *A, const int lda, float *packedA)
    {
        memset(packedA, 0, M * N * sizeof(float));
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                packedA[i * M + j] = A[j * lda + i];
            }
        }
    }

#if (SIMD_ARM_INSTR_SET >= SIMD_ARM7_NEON_VERSION)
    inline void cblas_sgemv_AnoTrans_neon(const int M, const int N, const float alpha, const float *A, const int lda,
                                          const float *x, const int incx, const float beta, float *y, const int incy)
    {
        const int restM = M % simd_registers;
        const int partM = M - restM;
        const int restN = N % mm_align_size;
        const int partN = N - restN;
        for (int i = 0; i < partM; i = i + simd_registers)
        {
            float32x4_t re[simd_registers] = {vdupq_n_f32(0.0f)};
            for (int j = 0; j < partN / mm_align_size; j++)
            {
                const int offset = j * mm_align_size;
                float32x4_t mx;
                if (incx == 1)
                {
                    mx = vld1q_f32(x + offset);
                }
                else
                {
                    mx = (float32x4_t){x[(offset + 3) * incx], x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]};
                }
                for (int ii = 0; ii < simd_registers; ii++)
                {
                    float32x4_t mA = vld1q_f32(A + (i + ii) * lda + offset);
#if (SIMD_ARM_INSTR_SET >= SIMD_ARM8_64_NEON_VERSION)
                    re[ii] = vfmaq_f32(re[ii], mA, mx);
#else
                    re[ii] = vmlaq_f32(re[ii], mA, mx);
#endif
                }
            }
            for (int ii = 0; ii < simd_registers; ii++)
            {
#if (SIMD_ARM_INSTR_SET >= SIMD_ARM8_64_NEON_VERSION)
                y[(i + ii) * incy] = alpha * vaddvq_f32(re[ii]) + beta * y[(i + ii) * incy];
#else
                float32x2_t _ss = vadd_f32(vget_low_f32(re[ii]), vget_high_f32(re[ii]));
                y[(i + ii) * incy] = alpha * vget_lane_f32(vpadd_f32(_ss, _ss), 0) + beta * y[(i + ii) * incy];
#endif
                const int A_offset = (i + ii) * lda;
                const int y_offset = (i + ii) * incy;
                for (int j = partN; j < N; j++)
                {
                    y[y_offset] += alpha * A[A_offset + j] * x[j * incx];
                }
            }
        }
        for (int i = partM; i < M; i++)
        {
            float32x4_t re = vdupq_n_f32(0.0f);
            for (int j = 0; j < partN / mm_align_size; j++)
            {
                const int offset = j * mm_align_size;
                float32x4_t mx;
                if (incx == 1)
                {
                    mx = vld1q_f32(x + offset);
                }
                else
                {
                    mx = (float32x4_t){x[(offset + 3) * incx], x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]};
                }
                float32x4_t mA = vld1q_f32(A + i * lda + offset);
#if (SIMD_ARM_INSTR_SET >= SIMD_ARM8_64_NEON_VERSION)
                re = vfmaq_f32(re, mA, mx);
#else
                re = vmlaq_f32(re, mA, mx);
#endif
            }
            const int A_offset = i * lda;
            const int y_offset = i * incy;
#if (SIMD_ARM_INSTR_SET >= SIMD_ARM8_64_NEON_VERSION)
            y[y_offset] = alpha * vaddvq_f32(re) + beta * y[y_offset];
#else
            float32x2_t _ss = vadd_f32(vget_low_f32(re), vget_high_f32(re));
            y[y_offset] = alpha * vget_lane_f32(vpadd_f32(_ss, _ss), 0) + beta * y[y_offset];
#endif
            for (int j = partN; j < N; j++)
            {
                y[y_offset] += alpha * A[A_offset + j] * x[j * incx];
            }
        }
    }
#endif

#if SIMD_TYPE == SIMDTYPE_SSE
    inline void cblas_sgemv_AnoTrans_sse(const int M, const int N, const float alpha, const float *A, const int lda,
                                         const float *x, const int incx, const float beta, float *y, const int incy)
    {
        const int restM = M % simd_registers;
        const int partM = M - restM;
        const int restN = N % mm_align_size;
        const int partN = N - restN;
        for (int i = 0; i < partM; i = i + simd_registers)
        {
            mm_type re[simd_registers] = {mm_setzero_ps()};
            for (int j = 0; j < partN / mm_align_size; j++)
            {
                const int offset = j * mm_align_size;
                mm_type mx;
                if (incx == 1)
                {
                    mx = mm_load_ps(x + offset);
                }
                else
                {
                    mx = _mm_set_ps(x[(offset + 3) * incx], x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]);
                }
                for (int ii = 0; ii < simd_registers; ii++)
                {
                    mm_type mA = mm_load_ps(A + (i + ii) * lda + offset);
                    re[ii] = mm_fmadd_ps(mA, mx, re[ii]);
                }
            }
            for (int ii = 0; ii < simd_registers; ii++)
            {
                y[(i + ii) * incy] = alpha * _mm_sumall_ps(re[ii]) + beta * y[(i + ii) * incy];
                const int A_offset = (i + ii) * lda;
                const int y_offset = (i + ii) * incy;
                for (int j = partN; j < N; j++)
                {
                    y[y_offset] += alpha * A[A_offset + j] * x[j * incx];
                }
            }
        }
        for (int i = partM; i < M; i++)
        {
            mm_type re = mm_setzero_ps();
            for (int j = 0; j < partN / mm_align_size; j++)
            {
                const int offset = j * mm_align_size;
                mm_type mx;
                if (incx == 1)
                {
                    mx = mm_load_ps(x + offset);
                }
                else
                {
                    mx = _mm_set_ps(x[(offset + 3) * incx], x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]);
                }
                mm_type mA = mm_load_ps(A + i * lda + offset);
                re = mm_fmadd_ps(mA, mx, re);
            }
            const int A_offset = i * lda;
            const int y_offset = i * incy;
            y[y_offset] = alpha * _mm_sumall_ps(re) + beta * y[y_offset];
            for (int j = partN; j < N; j++)
            {
                y[y_offset] += alpha * A[A_offset + j] * x[j * incx];
            }
        }
    }
#endif

#if SIMD_TYPE >= SIMDTYPE_AVX
    inline void cblas_sgemv_AnoTrans_avx(const int M, const int N, const float alpha, const float *A, const int lda,
                                         const float *x, const int incx, const float beta, float *y, const int incy)
    {
        const int restM = M % simd_registers;
        const int partM = M - restM;
        const int restN = N % mm_align_size;
        const int partN = N - restN;
        for (int i = 0; i < partM; i = i + simd_registers)
        {
            mm_type re[simd_registers] = {mm_setzero_ps()};
            for (int j = 0; j < partN / mm_align_size; j++)
            {
                const int offset = j * mm_align_size;
                mm_type mx;
                if (incx == 1)
                {
                    mx = mm_load_ps(x + offset);
                }
                else
                {
                    mx = _mm256_set_ps(x[(offset + 7) * incx], x[(offset + 6) * incx],
                                       x[(offset + 5) * incx], x[(offset + 4) * incx], x[(offset + 3) * incx],
                                       x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]);
                }
                for (int ii = 0; ii < simd_registers; ii++)
                {
                    mm_type mA = mm_load_ps(A + (i + ii) * lda + offset);
                    re[ii] = mm_fmadd_ps(mA, mx, re[ii]);
                }
            }
            float q[simd_registers];
            for (int ii = 0; ii < simd_registers; ii++)
            {
                y[(i + ii) * incy] = alpha * _mm256_sumall_ps(re[ii]) + beta * y[(i + ii) * incy];
                const int A_offset = (i + ii) * lda;
                const int y_offset = (i + ii) * incy;
                for (int j = partN; j < N; j++)
                {
                    y[y_offset] += alpha * A[A_offset + j] * x[j * incx];
                }
            }
        }
        for (int i = partM; i < M; i++)
        {
            mm_type re = mm_setzero_ps();
            for (int j = 0; j < partN / mm_align_size; j++)
            {
                const int offset = j * mm_align_size;
                mm_type mx;
                if (incx == 1)
                {
                    mx = mm_load_ps(x + offset);
                }
                else
                {
                    mx = _mm256_set_ps(x[(offset + 7) * incx], x[(offset + 6) * incx],
                                       x[(offset + 5) * incx], x[(offset + 4) * incx], x[(offset + 3) * incx],
                                       x[(offset + 2) * incx], x[(offset + 1) * incx], x[(offset + 0) * incx]);
                }
                mm_type mA = mm_load_ps(A + i * lda + offset);
                re = mm_fmadd_ps(mA, mx, re);
            }
            const int A_offset = i * lda;
            const int y_offset = i * incy;
            y[y_offset] = alpha * _mm256_sumall_ps(re) + beta * y[y_offset];
            for (int j = partN; j < N; j++)
            {
                y[y_offset] += alpha * A[A_offset + j] * x[j * incx];
            }
        }
    }
#endif

    void cblas_sgemv_AnoTrans(const int M, const int N, const float alpha, const float *A, const int lda,
                              const float *x, const int incx, const float beta, float *y, const int incy)
    {
#if SIMD_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
        NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_AVX
        cblas_sgemv_AnoTrans_avx(M, N, alpha, A, lda, x, incx, beta, y, incy);
#elif SIMD_TYPE >= SIMDTYPE_SSE
        cblas_sgemv_AnoTrans_sse(M, N, alpha, A, lda, x, incx, beta, y, incy);
#elif defined(__ARM_NEON)
        cblas_sgemv_AnoTrans_neon(M, N, alpha, A, lda, x, incx, beta, y, incy);
#else
#define UNHANDLED
        NATIVE_CODE_WARNING;
#endif
#ifdef UNHANDLED
        // Fall back to native code
        for (int i = 0; i < M; i++)
        {
            y[i * incy] = beta * y[i * incy];
            for (int j = 0; j < N; j++)
            {
                y[i * incy] += alpha * A[i * lda + j] * x[j * incx];
            }
        }
#undef UNHANDLED
#endif
    }

    // Convert this scenario into noTrans situation;
    void cblas_sgemv_ATrans(const int M, const int N, const float alpha, const float *A, const int lda,
                            const float *x, const int incx, const float beta, float *y, const int incy)
    {
#if SIMD_TYPE >= SIMDTYPE_AVX512
#define UNHANDLED
        NATIVE_CODE_WARNING;
#elif SIMD_TYPE >= SIMDTYPE_AVX
        //#define UNHANDLED
        float *packedA = new float[M * N];
        packTransedA(M, N, A, lda, packedA);
        cblas_sgemv_AnoTrans_avx(N, M, alpha, packedA, M, x, incx, beta, y, incy);
        delete[] packedA;
#elif SIMD_TYPE >= SIMDTYPE_SSE
        float *packedA = new float[M * N];
        packTransedA(M, N, A, lda, packedA);
        cblas_sgemv_AnoTrans_sse(N, M, alpha, packedA, M, x, incx, beta, y, incy);
        delete[] packedA;
#elif (SIMD_ARM_INSTR_SET >= SIMD_ARM7_NEON_VERSION)
        float *packedA = new float[M * N];
        packTransedA(M, N, A, lda, packedA);
        cblas_sgemv_AnoTrans_neon(N, M, alpha, packedA, M, x, incx, beta, y, incy);
        delete[] packedA;
#else
#define UNHANDLED
        // NATIVE_CODE_WARNING;
#endif
#ifdef UNHANDLED
        // Fall back to native code
        for (size_t j = 0; j < N; j++)
        {
            y[j * incy] = beta * y[j * incy];
            for (size_t i = 0; i < M; i++)
            {
                y[j * incy] += alpha * A[i * lda + j] * x[i * incx];
            }
        }
#undef UNHANDLED
#endif
    }
}
