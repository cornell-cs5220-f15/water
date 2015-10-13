#include <math.h>
#include <assert.h>

typedef float real;

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
