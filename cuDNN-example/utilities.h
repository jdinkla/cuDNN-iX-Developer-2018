/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */
#pragma once

#include "cuda_runtime.h"
#include "cudnn.h"
#include <stdio.h>
#include <cstdlib>

// inspired from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define check_cuda() { check_cuda_function(cudaGetLastError(), __FILE__, __LINE__); }
inline void check_cuda_function(const cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "check_cuda: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define check_cudnn(code) { check_cudnn_function(code, __FILE__, __LINE__); }
inline void check_cudnn_function(const cudnnStatus_t code, const char *file, int line, bool abort = true)
{
    if (code != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr, "check_cudnn: %s %s %d\n", cudnnGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
