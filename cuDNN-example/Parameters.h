/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */
#pragma once

struct Parameters {

    int tensor_w;
    int tensor_h;
    int tensor_n;
    int tensor_c;

    int filter_w;
    int filter_h;
    int filter_k;
    int filter_c;

    int padding_h;
    int padding_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;

    int out_n;
    int out_c;
    int out_h;
    int out_w;

    inline int get_tensor_bytes() {
        return tensor_n * tensor_c * tensor_h * tensor_w * sizeof(float);
    }

    inline int get_filter_bytes() {
        return filter_k * filter_c * filter_h * filter_w * sizeof(float);
    }

};