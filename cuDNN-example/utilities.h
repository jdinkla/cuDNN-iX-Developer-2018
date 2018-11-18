#pragma once

#include "cuda_runtime.h"
#include "cudnn.h"
#include <stdio.h>
#include <cstdlib>

// taken from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define check_cuda() { gpuAssert(__FILE__, __LINE__); }
inline void gpuAssert(const char *file, int line, bool abort = true)
{
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define check(code) { gpuAssert(code, __FILE__, __LINE__); }
inline void gpuAssert(const cudnnStatus_t code, const char *file, int line, bool abort = true)
{
    if (code != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudnnGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
