/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */
#pragma once

#include <iostream>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudnn.h"
#include "utilities.h"
#include "Parameters.h"

using namespace std;

class CudnnExample {

    Parameters params;

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t xDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t yDesc;

    cudnnConvolutionFwdAlgo_t algo;

    void* d_workspace = nullptr;
    size_t workspace_bytes;

    float* d_image = nullptr;
    float* d_w = nullptr;
    float* d_y = nullptr;

    int image_bytes;
    int w_bytes;
    int y_bytes;

    const float alpha = 1, beta = 0;

    // TODO
    int n = 1;
    int c = 1;
    int k = 1;

public:

    CudnnExample(Parameters& params) {
        this->params = params;

        image_bytes = n * c * params.w * params.h * sizeof(float);
        w_bytes = params.fw * params.fh * sizeof(float);
        y_bytes = params.fw * params.fh * sizeof(float);
    }

    void run() {
        cout << "Allocating CUDA and CUDNN ressources" << endl;
        allocateCuda();
        allocateCudnn();  

        cout << "Initializing convolution" << endl;
        setUpCudnn();
        determineBestForwardConvolution();
        std::cout << "The used algorithm is " << algo << std::endl;

        setUpWorkspace();

        cout << "Running the convolution" << endl;
        runAlgorithm();

        cout << "Freeing up the CUDA and CUDNN ressources" << endl;
        freeCudnn();
        freeCuda();
    }

protected:

    void allocateCuda();

    void allocateCudnn() {
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

    void setUpCudnn() {
        cudnnStatus_t result = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, params.w, params.h);
        check_cudnn(result);

        result = cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, params.fw, params.fh);
        check_cudnn(result);

        result = cudnnSetConvolution2dDescriptor(
            convDesc,
            0, //                             pad_h,
            0, //                             pad_w,
            2, //                             u,
            2, // int                             v,
            params.w, //                           dilation_h,
            params.h, // int                             dilation_w,
            CUDNN_CONVOLUTION,
            CUDNN_DATA_FLOAT);
        check_cudnn(result);

        result = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, k, c, params.fw, params.fh);
        check_cudnn(result);
    }

    void determineBestForwardConvolution() {
        cudnnStatus_t result;
        int convolutionForwardAlgorithmMaxCount = 0;
        result = cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &convolutionForwardAlgorithmMaxCount);
        check_cudnn(result);
        printf("convolutionForwardAlgorithmMaxCount = %d\n", convolutionForwardAlgorithmMaxCount);

        const int requestedAlgoCount = 3;
        cudnnConvolutionFwdAlgoPerf_t perfResults[requestedAlgoCount];
        printf("requestedAlgoCount = %d\n", requestedAlgoCount);

        int returnedAlgoCount;
        result = cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, &returnedAlgoCount, (cudnnConvolutionFwdAlgoPerf_t*)&perfResults);
        check_cudnn(result);
        printf("returnedAlgoCount = %d\n", returnedAlgoCount);

        for (int i = 0; i < returnedAlgoCount; i++)
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
        algo = perfResults[0].algo;
    }

    void setUpWorkspace() {
        workspace_bytes = 0;
        cudnnStatus_t result = cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, &workspace_bytes);
        check_cudnn(result);

        printf(" workspace_bytes = %zd\n", workspace_bytes);
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

    void runAlgorithm() {
        
        cudnnStatus_t result = cudnnConvolutionForward(
            handle,
            &alpha,
            xDesc,
            d_image,
            wDesc,
            d_w,
            convDesc,
            algo,
            d_workspace,
            workspace_bytes,
            &beta,
            yDesc,
            d_y);
        check_cudnn(result);
    }

    void freeCudnn() {
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

    void freeCuda();

};