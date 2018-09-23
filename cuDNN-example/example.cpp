#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudnn.h"
#include "utilities.h"
#include <stdio.h>

using namespace std;

int cuDNN() {
    const int w = 1024;
    const int h = 1024;
    cudnnStatus_t result;
    cudnnHandle_t handle;
    result = cudnnCreate(&handle);
    check(result);

    std::cout << "Got handle" << std::endl;

    int convolutionForwardAlgorithmMaxCount = 0;
    result = cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &convolutionForwardAlgorithmMaxCount);
    check(result);
    printf("convolutionForwardAlgorithmMaxCount = %d\n", convolutionForwardAlgorithmMaxCount);

    cudnnTensorDescriptor_t xDesc;
    result = cudnnCreateTensorDescriptor(&xDesc);
    check(result);

    result = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, w, h);
    check(result);

    cudnnFilterDescriptor_t wDesc;
    result = cudnnCreateFilterDescriptor(&wDesc);
    check(result);

    result = cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 5, 5);
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
        w, //                           dilation_h,
        h, // int                             dilation_w,
        CUDNN_CONVOLUTION,
        CUDNN_DATA_FLOAT);
    check(result);

    cudnnTensorDescriptor_t yDesc;
    result = cudnnCreateTensorDescriptor(&yDesc);
    check(result);

    result = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, w, h);
    check(result);

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

    const cudnnConvolutionFwdAlgo_t algorithm = perfResults[0].algo;
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

    const int image_bytes = 1 * 1 * 1024 * 1024 * sizeof(float);

    float* d_image = nullptr;
    float* d_w = nullptr;
    float* d_y = nullptr;
    // cudaMalloc(&d_input, image_bytes);
    // check_cuda();

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


    result = cudnnDestroy(handle);
    check(result);

    return 0;
}
