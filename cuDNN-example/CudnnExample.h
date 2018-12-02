/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */
#pragma once

#include <iostream>

#include "cudnn.h"
#include "Parameters.h"

using namespace std;

#define REQUESTED_NUMBER_OF_ALGORITHMS 3

class CudnnExample
{

private:

    cudnnHandle_t handle;
    cudnnTensorDescriptor_t xDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t yDesc;

    cudnnConvolutionFwdAlgo_t algo;

    void* d_workspace = nullptr;
    size_t workspace_bytes = 0;

    float* d_image = nullptr;
    float* d_w = nullptr;
    float* d_y = nullptr;

    Parameters params;

public:

    CudnnExample(Parameters& params)
    {
        this->params = params;
    }

    void run()
    {
        cout << "*** allocating CUDA and CUDNN ressources" << endl;
        allocateCuda();
        allocateCudnn();  

        cout << "*** initializing convolution" << endl;
        setUpCudnn();

        cout << "*** fill with test data" << endl;
        fillWithExampleData();

        determineBestForwardConvolution();
        std::cout << "The used algorithm is " << algo << std::endl;

        setUpWorkspace();

        cout << "*** running the convolution" << endl;
        runAlgorithm();

        reportResult();

        cout << "*** freeing up CUDA and CUDNN ressources" << endl;
        freeCudnn();
        freeCuda();
    }

protected:

    void allocateCuda();

    void allocateCudnn();

    void fillWithExampleData();

    void setUpCudnn();

    void determineBestForwardConvolution();

    void setUpWorkspace();

    void runAlgorithm();

    void freeCudnn();

    void freeCuda();

    void reportResult();
};