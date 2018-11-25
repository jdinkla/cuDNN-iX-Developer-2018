/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */

#include "CudnnExample.h"

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

void CudnnExample::allocateCuda() {
    cudaMalloc(&d_image, params.get_tensor_bytes());
    check_cuda();

    cudaMalloc(&d_w, params.get_filter_bytes());
    check_cuda();

    cudaMalloc(&d_y, params.get_filter_bytes());
    check_cuda();
}

void CudnnExample::freeCuda() {
    cudaFree(d_image);
    check_cuda();

    cudaFree(d_w);
    check_cuda();

    cudaFree(d_y);
    check_cuda();
}
