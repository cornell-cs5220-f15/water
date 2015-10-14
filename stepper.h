#ifndef STEPPER_H
#define STEPPER_H

#include <math.h>

typedef void (*flux_t)(float* FU, float* GU, const float* U,
                       int ncell, int field_stride);
typedef void (*speed_t)(float* cxy, const float* U,
                        int ncell, int field_stride);

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
void central2d_step(float* u, float* v, float* ux, float* uy,
                    float* f, float* fx, float* g, float* gx,
                    int io, int nx, int ny, int ng,
                    int nfield, flux_t flux, speed_t speed,
                    float dt, float dx, float dy);
void central2d_run(float* u, float* v, float* ux, float* uy,
                   float* f, float* fx, float* g, float* gy,
                   int nx, int ny, int ng,
                   int nfield, flux_t flux, speed_t speed,
                   float tfinal, float dx, float dy, float cfl);

#endif /* STEPPER_H */
