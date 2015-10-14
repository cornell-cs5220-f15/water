#include <string.h>
#include <math.h>

//ldoc on
/**
 * # Shallow water equations
 *
 * ## Physics picture
 *
 * The shallow water equations treat water as incompressible and
 * inviscid, and assume that the horizontal velocity remains constant
 * in any vertical column of water.  The unknowns at each point are
 * the water height and the total horizontal momentum in a water
 * column; the equations describe conservation of mass (fluid is
 * neither created nor destroyed) and conservation of linear momentum.
 * We will solve these equations with a numerical method that also
 * exactly conserves mass and momentum (up to rounding error), though
 * it only approximately conserves energy.
 *
 * The basic variables are water height ($h$), and the velocity components
 * ($u, v$).  We write the governing equations in the form
 * $$
 *   U_t = F(U)_x + G(U)_y
 * $$
 * where
 * $$
 *   U = \begin{bmatrix} h \\ hu \\ hv \end{bmatrix},
 *   F = \begin{bmatrix} hu \\ h^2 u + gh^2/2 \\ huv \end{bmatrix}
 *   G = \begin{bmatrix} hv \\ huv \\ h^2 v + gh^2/2 \end{bmatrix}
 * $$
 * The functions $F$ and $G$ are called *fluxes*, and describe how the
 * conserved quantities (volume and momentum) enter and exit a region
 * of space.
 *
 * Note that we also need a bound on the characteristic wave speeds
 * for the problem in order to ensure that our method doesn't explode;
 * we use this to control the Courant-Friedrichs-Levy (CFL) number
 * relating wave speeds, time steps, and space steps.  For the shallow
 * water equations, the characteristic wave speed is $\sqrt{g h}$
 * where $g$ is the gravitational constant and $h$ is the height of the
 * water; in addition, we have to take into account the velocity of
 * the underlying flow.
 *
 * ## Implementation
 *
 * Our solver takes advantage of C++ templates to get (potentially)
 * good performance while keeping a clean abstraction between the
 * solver code and the details of the physics.  The `Shallow2D`
 * class specifies the precision of the comptutation (single precision),
 * the data type used to represent vectors of unknowns and fluxes
 * (the C++ `std::array`).  We are really only using the class as
 * name space; we never create an instance of type `Shallow2D`,
 * and the `flux` and `wave_speed` functions needed by the solver are
 * declared as static (and inline, in the hopes of getting the compiler
 * to optimize for us).
 *
 * At the same time, it turns out that the C99 restrict keyword isn't
 * really supported in C++, so it probably makes sense to try for the
 * best of both worlds.
 */


static const float g = 9.8;


static
void shallow2dv_flux(float* restrict fh,
                     float* restrict fhu,
                     float* restrict fhv,
                     float* restrict gh,
                     float* restrict ghu,
                     float* restrict ghv,
                     const float* restrict h,
                     const float* restrict hu,
                     const float* restrict hv,
                     float g,
                     int ncell)
{
    memcpy(fh, hu, ncell * sizeof(float));
    memcpy(gh, hv, ncell * sizeof(float));
    for (int i = 0; i < ncell; ++i) {
        float hi = h[i], hui = hu[i], hvi = hv[i];
        float inv_h = 1/hi;
        fhu[i] = hui*hui*inv_h + (0.5*g)*hi*hi;
        fhv[i] = hui*hvi*inv_h;
        ghu[i] = hui*hvi*inv_h;
        ghv[i] = hvi*hvi*inv_h + (0.5*g)*hi*hi;
    }
}


static
void shallow2dv_speed(float* restrict cxy,
                      const float* restrict h,
                      const float* restrict hu,
                      const float* restrict hv,
                      float g,
                      int ncell)
{
    float cx = cxy[0];
    float cy = cxy[1];
    for (int i = 0; i < ncell; ++i) {
        float hi = h[i];
        float inv_hi = 1.0f/h[i];
        float root_gh = sqrtf(g * hi);
        float cxi = fabsf(hu[i] * inv_hi) + root_gh;
        float cyi = fabsf(hv[i] * inv_hi) + root_gh;
        cx = fmax(cx, cxi);
        cy = fmax(cy, cyi);
    }
    cxy[0] = cx;
    cxy[1] = cy;
}


// Compute shallow water fluxes F(U), G(U)
void shallow2d_flux(float* FU, float* GU, const float* U,
                    int ncell, int field_stride)
{
    shallow2dv_flux(FU, FU+field_stride, FU+2*field_stride,
                    GU, GU+field_stride, GU+2*field_stride,
                    U,  U +field_stride, U +2*field_stride,
                    g, ncell);
}


// Compute shallow water wave speed
void shallow2d_speed(float* cxy, const float* U,
                     int ncell, int field_stride)
{
    shallow2dv_speed(cxy, U, U+field_stride, U+2*field_stride, g, ncell);
}
