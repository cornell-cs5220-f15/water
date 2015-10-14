#ifndef STEPPER_H
#define STEPPER_H

#include <math.h>

extern "C" {
    void central2d_derivs(float* ux, float* uy, float* fx, float* gy,
                          const float* u, const float* f, const float* g,
                          int nx, int ny, int nfield);

    void central2d_periodic(float* u, int nx, int ny, int ng, int nfield);

    void central2d_predict(float* v, const float* u,
                           const float* fx, const float* gy,
                           float dtcdx2, float dtcdy2,
                           int nx, int ny, int nfield);
    void central2d_correct(float* v, const float* u,
                           const float* ux, const float* uy,
                           const float* f, const float* g,
                           float dtcdx2, float dtcdy2,
                           int xlo, int xhi, int ylo, int yhi,
                           int nx, int ny, int nfield);
}

//ldoc off
#endif /* STEPPER_H */
