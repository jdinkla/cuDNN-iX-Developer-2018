#pragma once

#include "Parameters.h"
#include "cuda_runtime.h"
#include "cudnn.h"

int cuDNN(Parameters& p);

cudnnConvolutionFwdAlgo_t determineBestForwardConvolution(
    const cudnnHandle_t &handle,
    const cudnnTensorDescriptor_t &xDesc,
    const cudnnFilterDescriptor_t &wDesc,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnTensorDescriptor_t &yDesc);
