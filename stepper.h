#ifndef STEPPER_H
#define STEPPER_H

#include <math.h>

typedef void (*flux_t)(float* FU, float* GU, const float* U,
                       int ncell, int field_stride);
typedef void (*speed_t)(float* cxy, const float* U,
                        int ncell, int field_stride);

typedef struct central2d_t {

    int nfield;   // Number of components in system
    int nx, ny;   // Grid resolution in x/y (without ghost cells)
    int ng;       // Number of ghost cells
    float dx, dy; // Cell width in x/y
    float cfl;    // Max allowed CFL number

    // Flux and speed functions
    flux_t flux;
    speed_t speed;

    // Storage
    float* u;
    float* v;
    float* ux;
    float* uy;
    float* f;
    float* fx;
    float* g;
    float* gy;

} central2d_t;


central2d_t* central2d_init(float w, float h, int nx, int ny,
                            int nfield, flux_t flux, speed_t speed,
                            float cfl);
void central2d_free(central2d_t* sim);
int  central2d_offset(central2d_t* sim, int k, int ix, int iy);
void central2d_run(central2d_t* sim, float tfinal);

void central2d_periodic(float* u, int nx, int ny, int ng, int nfield);

#endif /* STEPPER_H */
