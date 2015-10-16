#ifndef SHALLOW2D_VEC_H
#define SHALLOW2D_VEC_H

#include <array>
#include <cmath>

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
 * solver code and the details of the physics.  The `Shallow2DVec`
 * class specifies the precision of the comptutation (single precision),
 * the data type used to represent vectors of unknowns and fluxes
 * (the C++ `std::array`).  We are really only using the class as
 * name space; we never create an instance of type `Shallow2DVec`,
 * and the `flux` and `wave_speed` functions needed by the solver are
 * declared as static (and inline, in the hopes of getting the compiler
 * to optimize for us).
 */

struct Shallow2DVec {
    // Type parameters for solver
    typedef float real;
    typedef std::array<real,3> vec;

    // Number of fields in U vector
    static constexpr int num_fields = 3;

    // Gravitational force (compile time constant)
    static constexpr real g = 9.8;

    // Compute shallow water fluxes F(U), G(U)
    static void flux(real& FU0, real& FU1, real& FU2,
                     real& GU0, real& GU1, real& GU2,
                     const real h, const real hu, const real hv) {
        FU0 = hu;
        FU1 = hu*hu/h + (0.5f*g)*h*h;
        FU2 = hu*hv/h;

        GU0 = hv;
        GU1 = hu*hv/h;
        GU2 = hv*hv/h + (0.5f*g)*h*h;
    }

    // Compute shallow water wave speed
    static void wave_speed(real& cx, real& cy,
                           const real h, const real hu, const real hv) {
        using namespace std;
        real root_gh = sqrt(g * h);  // NB: Don't let h go negative!
        cx = abs(hu/h) + root_gh;
        cy = abs(hv/h) + root_gh;
    }
};

//ldoc off
#endif /* SHALLOW2D_VEC_H */
