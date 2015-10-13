#ifndef MINMOD_H
#define MINMOD_H

#include <math.h>

extern "C" {
    void limited_deriv1(float* du, float* u, int ncell);
    void limited_derivk(float* du, float* u, int ncell, int stride);
}

//ldoc off
#endif /* MINMOD_H */
