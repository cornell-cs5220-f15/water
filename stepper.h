#ifndef STEPPER_H
#define STEPPER_H

#include <math.h>

extern "C" {
    void limited_deriv1(float* du, float* u, int ncell);
    void limited_derivk(float* du, float* u, int ncell, int stride);
    void central2d_derivs(float* ux, float* uy, float* fx, float* gy,
                          const float* u, const float* f, const float* g,
                          int nx, int ny, int nfield);
}

//ldoc off
#endif /* STEPPER_H */
