#ifndef STEPPER_H
#define STEPPER_H

#include <math.h>

typedef void (*flux_t)(float* FU, float* GU, const float* U,
                       int ncell, int field_stride);
typedef void (*speed_t)(float* cxy, const float* U,
                        int ncell, int field_stride);

void central2d_periodic(float* u, int nx, int ny, int ng, int nfield);
void central2d_run(float* u, float* v, float* ux, float* uy,
                   float* f, float* fx, float* g, float* gy,
                   int nx, int ny, int ng,
                   int nfield, flux_t flux, speed_t speed,
                   float tfinal, float dx, float dy, float cfl);

#endif /* STEPPER_H */
