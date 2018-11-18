/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */
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
