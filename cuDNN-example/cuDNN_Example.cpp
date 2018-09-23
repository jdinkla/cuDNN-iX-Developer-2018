#include <iostream>

#include "utilities.h"

using namespace std;

extern int cuDNN();

int main(int argc, char** argv) {

    std::cout << "Hi" << std::endl;

    cuDNN();
    return 0;
}
