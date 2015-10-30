#ifndef iters
#define iters 100000000
#endif
#ifndef fiters
#define fiters 1
#endif

#include <math.h>

double fn(double x) {
    double result = x;
    for (int i = 0; i < fiters; i++) {
        result = cos(result + i);
    }
    return result;
}

