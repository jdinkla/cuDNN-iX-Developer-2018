/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */

#include <iostream>

#include "CudnnExample.h"
#include "utilities.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

void CudnnExample::allocateCudnn() {
    cudnnStatus_t result = cudnnCreate(&handle);
    check_cudnn(result);

    result = cudnnCreateTensorDescriptor(&xDesc);
    check_cudnn(result);

    result = cudnnCreateFilterDescriptor(&wDesc);
    check_cudnn(result);

    result = cudnnCreateConvolutionDescriptor(&convDesc);
    check_cudnn(result);

    result = cudnnCreateTensorDescriptor(&yDesc);
    check_cudnn(result);
}

void CudnnExample::allocateCuda() {
    cudaMalloc(&d_image, params.get_tensor_bytes());
    check_cuda();

    cudaMalloc(&d_w, params.get_filter_bytes());
    check_cuda();

    cudaMalloc(&d_y, params.get_filter_bytes());
    check_cuda();
}

void CudnnExample::determineBestForwardConvolution() {
    cudnnStatus_t result;
    int convolutionForwardAlgorithmMaxCount = 0;
    result = cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &convolutionForwardAlgorithmMaxCount);
    check_cudnn(result);
    cout << "convolutionForwardAlgorithmMaxCount = " << convolutionForwardAlgorithmMaxCount << endl;

    cudnnConvolutionFwdAlgoPerf_t algorithms[REQUESTED_NUMBER_OF_ALGORITHMS];
    cout << "requested_number_of_algorithms = " << REQUESTED_NUMBER_OF_ALGORITHMS << endl;

    int returned_number_of_algorithms;
    result = cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc,
        REQUESTED_NUMBER_OF_ALGORITHMS, &returned_number_of_algorithms, (cudnnConvolutionFwdAlgoPerf_t*)&algorithms);
    check_cudnn(result);
    cout << "returned_number_of_algorithms = " << returned_number_of_algorithms << endl;

    for (int i = 0; i < returned_number_of_algorithms; i++)
    {
        cudnnConvolutionFwdAlgoPerf_t current = algorithms[i];
        cout << "---- " << i << endl
            << " algo " << current.algo << endl
            << " status " << current.status << endl
            << " time " << current.time << endl
            << " memory " << current.memory << endl
            << " determinism " << current.determinism << endl
            << " mathType " << current.mathType << endl;
    }
    algo = algorithms[0].algo;
}

void CudnnExample::freeCuda() {
    cudaFree(d_image);
    check_cuda();

    cudaFree(d_w);
    check_cuda();

    cudaFree(d_y);
    check_cuda();
}

void CudnnExample::freeCudnn() {
    cudnnStatus_t result = cudnnDestroyConvolutionDescriptor(convDesc);
    check_cudnn(result);

    result = cudnnDestroyTensorDescriptor(xDesc);
    check_cudnn(result);

    result = cudnnDestroyTensorDescriptor(yDesc);
    check_cudnn(result);

    result = cudnnDestroyFilterDescriptor(wDesc);
    check_cudnn(result);

    result = cudnnDestroy(handle);
    check_cudnn(result);

}

void CudnnExample::runAlgorithm() {
    cudnnStatus_t result = cudnnConvolutionForward(
        handle,
        &params.alpha,
        xDesc,
        d_image,
        wDesc,
        d_w,
        convDesc,
        algo,
        d_workspace,
        workspace_bytes,
        &params.beta,
        yDesc,
        d_y);
    check_cudnn(result);
}

void CudnnExample::setUpCudnn() {
    cudnnStatus_t result = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        params.tensor_n, params.tensor_c, params.tensor_h, params.tensor_w);
    check_cudnn(result);

    result = cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        params.filter_k, params.filter_c, params.filter_h, params.filter_w);
    check_cudnn(result);

    result = cudnnSetConvolution2dDescriptor(
        convDesc,
        params.padding_h,  //  pad_h,
        params.padding_w,  //  pad_w,
        params.stride_h,   //  u,
        params.stride_w,   //  v,
        params.dilation_h, //  dilation_h,
        params.dilation_w, //  dilation_w
        CUDNN_CONVOLUTION,
        CUDNN_DATA_FLOAT);
    check_cudnn(result);

    result = cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc,
        &params.out_n, &params.out_c, &params.out_h, &params.out_w);

    cout << "cudnnGetConvolution2dForwardOutputDim returned n= " << params.out_n
        << ", c=" << params.out_c << ", h=" << params.out_h << ", w=" << params.out_w << endl;

    result = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        params.out_n, params.out_c, params.out_h, params.out_w);
    check_cudnn(result);
}

void CudnnExample::setUpWorkspace() {
    workspace_bytes = 0;
    cudnnStatus_t result = cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, &workspace_bytes);
    check_cudnn(result);

    cout << "workspace_bytes = " << workspace_bytes << endl;
    cudnnStatus_t CUDNNWINAPI
        cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
            const cudnnTensorDescriptor_t xDesc,
            const cudnnFilterDescriptor_t wDesc,
            const cudnnConvolutionDescriptor_t convDesc,
            const cudnnTensorDescriptor_t yDesc,
            cudnnConvolutionFwdAlgo_t algo,
            size_t *sizeInBytes);

    cudaMalloc(&d_workspace, workspace_bytes);
}
