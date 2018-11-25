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

#define REQUESTED_NUMBER_OF_ALGORITHMS 3


class CudnnExample {

private:

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

public:

    const float alpha = 1, beta = 0;
    Parameters params;

    CudnnExample(Parameters& params) {
        this->params = params;
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

    void allocateCudnn();

    void setUpCudnn() {
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

    void determineBestForwardConvolution() {
        cudnnStatus_t result;
        int convolutionForwardAlgorithmMaxCount = 0;
        result = cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &convolutionForwardAlgorithmMaxCount);
        check_cudnn(result);
        printf("convolutionForwardAlgorithmMaxCount = %d\n", convolutionForwardAlgorithmMaxCount);
        
        cudnnConvolutionFwdAlgoPerf_t algorithms[REQUESTED_NUMBER_OF_ALGORITHMS];
        printf("requested_number_of_algorithms = %d\n", REQUESTED_NUMBER_OF_ALGORITHMS);

        int returned_number_of_algorithms;
        result = cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc,
            REQUESTED_NUMBER_OF_ALGORITHMS, &returned_number_of_algorithms, (cudnnConvolutionFwdAlgoPerf_t*)&algorithms);
        check_cudnn(result);
        printf("returned_number_of_algorithms = %d\n", returned_number_of_algorithms);

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

    void freeCudnn();

    void freeCuda();

};