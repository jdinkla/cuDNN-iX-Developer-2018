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
