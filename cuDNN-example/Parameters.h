/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */
#pragma once

struct Parameters
{

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

    float alpha;
    float beta;

    inline int get_tensor_bytes()
    {
        return tensor_n * tensor_c * tensor_h * tensor_w * sizeof(float);
    }

    inline int get_filter_bytes()
    {
        return filter_k * filter_c * filter_h * filter_w * sizeof(float);
    }

    inline int get_tensor_index(int n, int c, int h, int w)
    {
        const int sN = tensor_c * tensor_h * tensor_w;
        const int sC = tensor_h * tensor_w;
        const int sH = tensor_w;
        return n * sN + c * sC + h * sH + w;
    }

    inline int get_filter_index(int k, int c, int h, int w)
    {
        const int sK = filter_c * filter_h * filter_w;
        const int sC = filter_h * filter_w;
        const int sH = filter_w;
        return k * sK + c * sC + h * sH + w;
    }

    inline int get_out_index(int n, int c, int h, int w)
    {
        const int sN = out_c * out_h * out_w;
        const int sC = out_h * out_w;
        const int sH = out_w;
        return n * sN + c * sC + h * sH + w;
    }
};