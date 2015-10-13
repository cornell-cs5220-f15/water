#include <math.h>
#include <assert.h>

typedef float real;

/**
 * ### Derivatives with limiters
 *
 * In order to advance the time step, we also need to estimate
 * derivatives of the fluxes and the solution values at each cell.
 * In order to maintain stability, we apply a limiter here.
 */


// Branch-free computation of minmod of two numbers times 2s
static inline
real xmin2s(real s, real a, real b) {
    return ((copysignf(s, a) +
             copysignf(s, b)) *
            fminf( fabsf(a), fabsf(b) ));
}


// Limited combined slope estimate
static inline
real limdiff(real um, real u0, real up) {
    const real theta = 2.0;
    const real quarter = 0.25;
    real du1 = u0-um;   // Difference to left
    real du2 = up-u0;   // Difference to right
    real duc = up-um;   // Twice centered difference
    return xmin2s( quarter, xmin2s(theta, du1, du2), duc );
}


// Compute limited derivs
void limited_deriv1(real* restrict du,
                    const real* restrict u,
                    int ncell)
{
    for (int i = 0; i < ncell; ++i)
        du[i] = limdiff(u[i-1], u[i], u[i+1]);
}


// Compute limited derivs across stride
void limited_derivk(real* restrict du,
                    const real* restrict u,
                    int ncell, int stride)
{
    assert(stride > 0);
    for (int i = 0; i < ncell; ++i)
        du[i] = limdiff(u[i-stride], u[i], u[i+stride]);
}


// Compute limited derivs over grid
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
