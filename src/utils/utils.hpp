#ifndef UTILS_HPP
#define UTILS_HPP

#include "julius/julius.hpp"

#include <iostream>

namespace juliusblas
{
    template <typename T>
    void PrintMatrix(const enum CBLAS_LAYOUT order, int m, int n, T *ptr)
    {
        if (order == CblasRowMajor)
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    std::cout << ptr[i * m + j] << "\t"
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
                    std::cout << ptr[j * m + i] << "\t"
                }
                std::cout << std::endl;
            }
        }
        else
        {
            LOGE(" Errors in Julius print matrix.\n");
        }
    }
} // juliusblas

#endif