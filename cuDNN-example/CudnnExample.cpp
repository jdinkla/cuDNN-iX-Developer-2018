/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */

#include "CudnnExample.h"

void CudnnExample::allocateCuda() {
    cudaMalloc(&d_image, image_bytes);
    check_cuda();

    cudaMalloc(&d_w, w_bytes);
    check_cuda();

    cudaMalloc(&d_y, y_bytes);
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