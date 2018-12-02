/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */

#include "CudnnExample.h"
#include "Parameters.h"

int main(int argc, char** argv) {
    Parameters p;
    p.tensor_w = 4;
    p.tensor_h = 4;
    p.tensor_n = 1;
    p.tensor_c = 1;

    p.filter_w = 2;
    p.filter_h = 2;
    p.filter_k = 1;
    p.filter_c = 1;

    p.padding_h = 1;
    p.padding_w = 1;
    p.stride_h = 1;
    p.stride_w = 1;
    p.dilation_h = 1;
    p.dilation_w = 1;

    p.alpha = 1.0;
    p.beta = 0.0;

    CudnnExample cudnnExample(p);
    cudnnExample.run();
    return 0;
}
