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
    p.w = 16;
    p.h = 16;
    p.fw = 5;
    p.fh = 5;

    CudnnExample cudnnExample(p);
    cudnnExample.run();
    return 0;
}
