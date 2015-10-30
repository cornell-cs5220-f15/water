#ifndef SHALLOW2D_H
#define SHALLOW2D_H

#if defined _PARALLEL_DEVICE
#pragma offload_attribute(push,target(mic))
#endif
#include <cmath>
#include <array>
#if defined _PARALLEL_DEVICE
#pragma offload_attribute(pop)
#endif

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
 */

/* The following allows for minimal SIMD vectorization using GCC,
 * but at the very least allows local compilation before sending
 * to the cluster.
 */
#ifdef __INTEL_COMPILER
    #define DEF_ALIGN(x) __declspec(align((x)))
    #define USE_ALIGN(var, align) __assume_aligned((var), (align));
#else // GCC
    #define DEF_ALIGN(x) __attribute__ ((aligned((x))))
    #define USE_ALIGN(var, align) ((void)0) /* __builtin_assume_align is unreliabale... */
#endif

#if defined _PARALLEL_DEVICE
    #define TARGET_MIC __declspec(target(mic))
#else
    #define TARGET_MIC /* n/a */
#endif
struct Shallow2D {

    // global constants for alignment
    TARGET_MIC
    static constexpr int vec_size  = 4;
    TARGET_MIC
    static constexpr int VEC_ALIGN = 16;
    #if defined _PARALLEL_DEVICE
        TARGET_MIC
        static constexpr int BYTE_ALIGN = 64;
    #else
        static constexpr int BYTE_ALIGN = 32;
    #endif

    // Type parameters for solver
    typedef float real;
    typedef std::array<real, vec_size> vec;

    // Gravitational force (compile time constant)
    TARGET_MIC
    static constexpr real g = 9.8f;

    // Compute shallow water fluxes F(U), G(U)
    TARGET_MIC
    static inline void flux(real *FU, real *GU, const real *U) {
        USE_ALIGN(FU, VEC_ALIGN);
        USE_ALIGN(GU, VEC_ALIGN);
        USE_ALIGN(U , VEC_ALIGN);

        real h = U[0], hu = U[1], hv = U[2];

        FU[0] = hu;
        FU[1] = hu*hu/h + (0.5f*g)*h*h;
        FU[2] = hu*hv/h;

        GU[0] = hv;
        GU[1] = hu*hv/h;
        GU[2] = hv*hv/h + (0.5f*g)*h*h;
    }

    // Compute shallow water wave speed
    TARGET_MIC
    static inline void wave_speed(real& cx, real& cy, const real *U) {
        USE_ALIGN(U, VEC_ALIGN);

        using namespace std;
        real h = U[0], hu = U[1], hv = U[2];
        real root_gh = sqrt(g * h);  // NB: Don't let h go negative!
        cx = fabs(hu/h) + root_gh;
        cy = fabs(hv/h) + root_gh;
    }
};

//ldoc off
#endif /* SHALLOW2D_H */
