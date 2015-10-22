#include "stepper.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>

//ldoc on
/**
 * ## Implementation
 *
 * ### Structure allocation
 */

central2d_t* central2d_init(float w, float h, int nx, int ny,
                            int nfield, flux_t flux, speed_t speed,
                            float cfl)
{
    int ng = 3;

    central2d_t* sim = (central2d_t*) malloc(sizeof(central2d_t));
    sim->nx = nx;
    sim->ny = ny;
    sim->ng = ng;
    sim->nfield = nfield;
    sim->dx = w/nx;
    sim->dy = h/ny;
    sim->flux = flux;
    sim->speed = speed;
    sim->cfl = cfl;

    int N = nfield * (nx+2*ng) * (ny+2*ng);
    sim->u  = (float*) malloc(8 * N * sizeof(float));
    sim->v  = sim->u +   N;
    sim->ux = sim->u + 2*N;
    sim->uy = sim->u + 3*N;
    sim->f  = sim->u + 4*N;
    sim->fx = sim->u + 5*N;
    sim->g  = sim->u + 6*N;
    sim->gy = sim->u + 7*N;

    return sim;
}


void central2d_free(central2d_t* sim)
{
    free(sim->u);
    free(sim);
}


int central2d_offset(central2d_t* sim, int k, int ix, int iy)
{
    int nx = sim->nx, ny = sim->ny, ng = sim->ng;
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    return (k*ny_all+(ng+iy))*nx_all+(ng+ix);
}


/**
 * ### Boundary conditions
 *
 * In finite volume methods, boundary conditions are typically applied by
 * setting appropriate values in ghost cells.  For our framework, we will
 * apply periodic boundary conditions; that is, waves that exit one side
 * of the domain will enter from the other side.
 *
 * We apply the conditions by assuming that the cells with coordinates
 * `nghost <= ix <= nx+nghost` and `nghost <= iy <= ny+nghost` are
 * "canonical", and setting the values for all other cells `(ix,iy)`
 * to the corresponding canonical values `(ix+p*nx,iy+q*ny)` for some
 * integers `p` and `q`.
 */

static inline
void copy_subgrid(float* restrict dst,
                  const float* restrict src,
                  int nx, int ny, int stride)
{
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix)
            dst[iy*stride+ix] = src[iy*stride+ix];
}

void central2d_periodic(float* restrict u,
                        int nx, int ny, int ng, int nfield)
{
    // Stride and number per field
    int s = nx + 2*ng;
    int field_stride = (ny+2*ng)*s;

    // Offsets of left, right, top, and bottom data blocks and ghost blocks
    int l = nx,   lg = 0;
    int r = ng,   rg = nx+ng;
    int b = ny*s, bg = 0;
    int t = ng*s, tg = (nx+ng)*s;

    // Copy data into ghost cells on each side
    for (int k = 0; k < nfield; ++k) {
        float* uk = u + k*field_stride;
        copy_subgrid(uk+lg, uk+l, ng, ny+2*ng, s);
        copy_subgrid(uk+rg, uk+r, ng, ny+2*ng, s);
        copy_subgrid(uk+tg, uk+t, nx+2*ng, ng, s);
        copy_subgrid(uk+bg, uk+b, nx+2*ng, ng, s);
    }
}


/**
 * ### Derivatives with limiters
 *
 * In order to advance the time step, we also need to estimate
 * derivatives of the fluxes and the solution values at each cell.
 * In order to maintain stability, we apply a limiter here.
 *
 * The minmod limiter *looks* like it should be expensive to computer,
 * since superficially it seems to require a number of branches.
 * We do something a little tricky, getting rid of the condition
 * on the sign of the arguments using the `copysign` instruction.
 * If the compiler does the "right" thing with `max` and `min`
 * for floating point arguments (translating them to branch-free
 * intrinsic operations), this implementation should be relatively fast.
 */


// Branch-free computation of minmod of two numbers times 2s
static inline
float xmin2s(float s, float a, float b) {
    float sa = copysignf(s, a);
    float sb = copysignf(s, b);
    float abs_a = fabsf(a);
    float abs_b = fabsf(b);
    float min_abs = fminf(abs_a, abs_b);
    return (sa+sb) * min_abs;
}


// Limited combined slope estimate
static inline
float limdiff(float um, float u0, float up) {
    const float theta = 2.0;
    const float quarter = 0.25;
    float du1 = u0-um;   // Difference to left
    float du2 = up-u0;   // Difference to right
    float duc = up-um;   // Twice centered difference
    return xmin2s( quarter, xmin2s(theta, du1, du2), duc );
}


// Compute limited derivs
static inline
void limited_deriv1(float* restrict du,
                    const float* restrict u,
                    int ncell)
{
    for (int i = 0; i < ncell; ++i)
        du[i] = limdiff(u[i-1], u[i], u[i+1]);
}


// Compute limited derivs across stride
static inline
void limited_derivk(float* restrict du,
                    const float* restrict u,
                    int ncell, int stride)
{
    assert(stride > 0);
    for (int i = 0; i < ncell; ++i)
        du[i] = limdiff(u[i-stride], u[i], u[i+stride]);
}


// Compute limited derivs over grid
static
void central2d_derivs(float* restrict ux, float* restrict uy,
                      float* restrict fx, float* restrict gy,
                      const float* restrict u,
                      const float* restrict f,
                      const float* restrict g,
                      int nx, int ny, int nfield)
{
    for (int k = 0; k < nfield; ++k)
        for (int iy = 1; iy < ny-1; ++iy) {
            int offset = (k*ny+iy)*nx+1;
            limited_deriv1(ux+offset, u+offset, nx-2);
            limited_deriv1(fx+offset, f+offset, nx-2);
            limited_derivk(uy+offset, u+offset, nx-2, nx);
            limited_derivk(gy+offset, g+offset, nx-2, nx);
        }
}


/**
 * ### Advancing a time step
 *
 * Take one step of the numerical scheme.  This consists of two pieces:
 * a first-order corrector computed at a half time step, which is used
 * to obtain new $F$ and $G$ values; and a corrector step that computes
 * the solution at the full step.  For full details, we refer to the
 * [Jiang and Tadmor paper][jt].
 *
 * The `compute_step` function takes two arguments: the `io` flag
 * which is the time step modulo 2 (0 if even, 1 if odd); and the `dt`
 * flag, which actually determines the time step length.  We need
 * to know the even-vs-odd distinction because the Jiang-Tadmor
 * scheme alternates between a primary grid (on even steps) and a
 * staggered grid (on odd steps).  This means that the data at $(i,j)$
 * in an even step and the data at $(i,j)$ in an odd step represent
 * values at different locations in space, offset by half a space step
 * in each direction.  Every other step, we shift things back by one
 * mesh cell in each direction, essentially resetting to the primary
 * indexing scheme.
 */


// Predictor half-step
static
void central2d_predict(float* restrict v,
                       const float* restrict u,
                       const float* restrict fx,
                       const float* restrict gy,
                       float dtcdx2, float dtcdy2,
                       int nx, int ny, int nfield)
{
    for (int k = 0; k < nfield; ++k)
        for (int iy = 1; iy < ny-1; ++iy)
            for (int ix = 1; ix < nx-1; ++ix) {
                int offset = (k*ny+iy)*nx+ix;
                v[offset] = u[offset] -
                    dtcdx2 * fx[offset] -
                    dtcdy2 * gy[offset];
            }
}


// Corrector
static
void central2d_correct(float* restrict v,
                       const float* restrict u,
                       const float* restrict ux,
                       const float* restrict uy,
                       const float* restrict f,
                       const float* restrict g,
                       float dtcdx2, float dtcdy2,
                       int xlo, int xhi, int ylo, int yhi,
                       int nx, int ny, int nfield)
{
    assert(0 <= xlo && xlo < xhi && xhi <= nx);
    assert(0 <= ylo && ylo < yhi && yhi <= ny);

    for (int k = 0; k < nfield; ++k)
        for (int iy = ylo; iy < yhi; ++iy)
            for (int ix = xlo; ix < xhi; ++ix) {

                int j00 = (k*ny+iy)*nx+ix;
                int j10 = j00+1;
                int j01 = j00+nx;
                int j11 = j00+nx+1;

                v[j00] =
                    0.2500f * ( u[j00] + u[j01] + u[j10] + u[j11] ) -
                    0.0625f * ( ux[j10] - ux[j00] +
                                ux[j11] - ux[j01] +
                                uy[j01] - uy[j00] +
                                uy[j11] - uy[j10] ) -
                    dtcdx2  * ( f[j10] - f[j00] +
                                f[j11] - f[j01] ) -
                    dtcdy2  * ( g[j01] - g[j00] +
                                g[j11] - g[j10] );
            }
}


static
void central2d_step(float* restrict u, float* restrict v,
                    float* restrict ux,
                    float* restrict uy,
                    float* restrict f,
                    float* restrict fx,
                    float* restrict g,
                    float* restrict gy,
                    int io, int nx, int ny, int ng,
                    int nfield, flux_t flux, speed_t speed,
                    float dt, float dx, float dy)
{
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;

    float dtcdx2 = 0.5 * dt / dx;
    float dtcdy2 = 0.5 * dt / dy;

    flux(f, g, u, nx_all * ny_all, nx_all * ny_all);
    central2d_derivs(ux, uy, fx, gy, u, f, g,
                     nx_all, ny_all, nfield);
    central2d_predict(v, u, fx, gy, dtcdx2, dtcdy2,
                      nx_all, ny_all, nfield);

    // Flux values of f and g at half step
    for (int iy = 1; iy < ny_all-1; ++iy) {
        int jj = iy*nx_all+1;
        flux(f+jj, g+jj, v+jj, nx_all-2, nx_all * ny_all);
    }

    central2d_correct(v, u, ux, uy, f, g, dtcdx2, dtcdy2,
                      ng-io, nx+ng-io,
                      ng-io, ny+ng-io,
                      nx_all, ny_all, nfield);

    // Copy from v storage back to main grid
    for (int k = 0; k < nfield; ++k)
        memcpy(u+(k*ny_all+ng   )*nx_all+ng,
               v+(k*ny_all+ng-io)*nx_all+ng-io,
               ny * nx_all * sizeof(float));
}


/**
 * ### Advance a fixed time
 *
 * The `run` method advances from time 0 (initial conditions) to time
 * `tfinal`.  Note that `run` can be called repeatedly; for example,
 * we might want to advance for a period of time, write out a picture,
 * advance more, and write another picture.  In this sense, `tfinal`
 * should be interpreted as an offset from the time represented by
 * the simulator at the start of the call, rather than as an absolute time.
 *
 * We always take an even number of steps so that the solution
 * at the end lives on the main grid instead of the staggered grid.
 */

static
int central2d_xrun(float* restrict u, float* restrict v,
                   float* restrict ux,
                   float* restrict uy,
                   float* restrict f,
                   float* restrict fx,
                   float* restrict g,
                   float* restrict gy,
                   int nx, int ny, int ng,
                   int nfield, flux_t flux, speed_t speed,
                   float tfinal, float dx, float dy, float cfl)
{
    int nstep = 0;
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    bool done = false;
    float t = 0;
    while (!done) {
        float dt;
        for (int io = 0; io < 2; ++io) {
            float cxy[2] = {1.0e-15f, 1.0e-15f};
            central2d_periodic(u, nx, ny, ng, nfield);
            speed(cxy, u, nx_all * ny_all, nx_all * ny_all);
            if (io == 0) {
                dt = cfl / fmaxf(cxy[0]/dx, cxy[1]/dy);
                if (t + 2*dt >= tfinal) {
                    dt = (tfinal-t)/2;
                    done = true;
                }
            }
            central2d_step(u, v, ux, uy, f, fx, g, gy,
                           io, nx, ny, ng,
                           nfield, flux, speed,
                           dt, dx, dy);
            t += dt;
            nstep += 2;
        }
    }
    return nstep;
}


int central2d_run(central2d_t* sim, float tfinal)
{
   return central2d_xrun(sim->u, sim->v, sim->ux, sim->uy,
                         sim->f, sim->fx, sim->g, sim->gy,
                         sim->nx, sim->ny, sim->ng,
                         sim->nfield, sim->flux, sim->speed,
                         tfinal, sim->dx, sim->dy, sim->cfl);
}
