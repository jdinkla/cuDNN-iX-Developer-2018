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

    inline int get_tensor_bytes() {
        return tensor_n * tensor_c * tensor_h * tensor_w * sizeof(float);
    }

    inline int get_filter_bytes() {
        return filter_k * filter_c * filter_h * filter_w * sizeof(float);
    }

};