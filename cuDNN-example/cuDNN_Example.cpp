/*
 * Code for the article "Unter der Haube" from the german magazine iX-Developer "Maschinelles Lernen"
 *
 * See https://github.com/jdinkla/cuDNN-iX-Developer-2018
 *
 * (c) 2018 Jörn Dinkla, https://www.dinkla.net
 */
#include <iostream>

#include "utilities.h"
#include "example.h"
#include "Parameters.h"

using namespace std;

int main(int argc, char** argv) {

    std::cout << "Hi" << std::endl;

    Parameters p;
    p.w = 1024;
    p.h = 1024;
    p.fw = 5;
    p.fh = 5;

    cuDNN(p);
    return 0;
}
