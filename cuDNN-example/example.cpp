/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */
#include <stdio.h>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudnn.h"
#include "utilities.h"
#include "Parameters.h"

using namespace std;

cudnnConvolutionFwdAlgo_t determineBestForwardConvolution(
    const cudnnHandle_t &handle,
    const cudnnTensorDescriptor_t &xDesc,
    const cudnnFilterDescriptor_t &wDesc,
    const cudnnConvolutionDescriptor_t &convDesc,
    const cudnnTensorDescriptor_t &yDesc)
{
    cudnnStatus_t result;
    int convolutionForwardAlgorithmMaxCount = 0;
    result = cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &convolutionForwardAlgorithmMaxCount);
    check(result);
    printf("convolutionForwardAlgorithmMaxCount = %d\n", convolutionForwardAlgorithmMaxCount);

    const int requestedAlgoCount = 3;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[requestedAlgoCount];

    result = cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, &returnedAlgoCount, (cudnnConvolutionFwdAlgoPerf_t*)&perfResults);
    check(result);

    for (int i = 0; i < requestedAlgoCount; i++)
    {
        cudnnConvolutionFwdAlgoPerf_t current = perfResults[i];
        printf("---- %d\n", i);
        printf(" algo %d\n", current.algo);
        printf(" status %d\n", current.status);
        printf(" time %f\n", current.time);
        printf(" memory %zd\n", current.memory);
        printf(" determinism %d\n", current.determinism);
        printf(" mathType %d\n", current.mathType);
    }
    return perfResults[0].algo;
}

int cuDNN(Parameters& p) {

    const int n = 2;
    const int k = 1;
    const int c = 1;

    // Create a handle
    cudnnStatus_t result;
    cudnnHandle_t handle;
    result = cudnnCreate(&handle);
    check(result);
    std::cout << "Got handle" << std::endl;

    // Create the cuDNN resources
    cudnnTensorDescriptor_t xDesc;
    result = cudnnCreateTensorDescriptor(&xDesc);
    check(result);

    result = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, p.w, p.h);
    check(result);

    cudnnFilterDescriptor_t wDesc;
    result = cudnnCreateFilterDescriptor(&wDesc);
    check(result);

    result = cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, p.fw, p.fh);
    check(result);

    cudnnConvolutionDescriptor_t convDesc;
    result = cudnnCreateConvolutionDescriptor(&convDesc);
    check(result);

    result = cudnnSetConvolution2dDescriptor(
        convDesc,
        0, //                             pad_h,
        0, //                             pad_w,
        2, //                             u,
        2, // int                             v,
        p.w, //                           dilation_h,
        p.h, // int                             dilation_w,
        CUDNN_CONVOLUTION,
        CUDNN_DATA_FLOAT);
    check(result);

    cudnnTensorDescriptor_t yDesc;
    result = cudnnCreateTensorDescriptor(&yDesc);
    check(result);

    result = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, k, c, p.fw, p.fh);
    check(result);

    // Determine the algorithm
    const cudnnConvolutionFwdAlgo_t algorithm = determineBestForwardConvolution(handle, xDesc, wDesc, convDesc, yDesc);
    std::cout << "The used algorithm is " << algorithm << std::endl;

    size_t workspace_bytes = 0;
    result = cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algorithm, &workspace_bytes);
    check(result);

    printf(" workspace_bytes = %zd\n", workspace_bytes);

    cudnnStatus_t CUDNNWINAPI
        cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
            const cudnnTensorDescriptor_t xDesc,
            const cudnnFilterDescriptor_t wDesc,
            const cudnnConvolutionDescriptor_t convDesc,
            const cudnnTensorDescriptor_t yDesc,
            cudnnConvolutionFwdAlgo_t algo,
            size_t *sizeInBytes);

    void* d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_bytes);

    // Allocate memory on device
    const int image_bytes = n * c * p.w * p.h * sizeof(float);
    const int w_bytes = p.fw * p.fh * sizeof(float);
    const int y_bytes = p.fw * p.fh * sizeof(float);

    float* d_image = nullptr;
    cudaMalloc(&d_image, image_bytes);
    check_cuda();

    float* d_w = nullptr;
    cudaMalloc(&d_w, w_bytes);
    check_cuda();

    float* d_y = nullptr;
    cudaMalloc(&d_y, y_bytes);
    check_cuda();

    const float alpha = 1, beta = 0;
    result = cudnnConvolutionForward(
        handle,
        &alpha,
        xDesc,
        d_image,
        wDesc,
        d_w,
        convDesc,
        algorithm,
        d_workspace,
        workspace_bytes,
        &beta,
        yDesc,
        d_y);
    check(result);

    result = cudnnDestroyConvolutionDescriptor(convDesc);
    check(result);

    result = cudnnDestroyTensorDescriptor(xDesc);
    check(result);

    result = cudnnDestroyTensorDescriptor(yDesc);
    check(result);

    result = cudnnDestroyFilterDescriptor(wDesc);
    check(result);

    // destroy CUDA resources
    cudaFree(d_image);
    check_cuda();

    cudaFree(d_w);
    check_cuda();

    cudaFree(d_y);
    check_cuda();

    // Destroy the handle
    result = cudnnDestroy(handle);
    check(result);

    return 0;
}
