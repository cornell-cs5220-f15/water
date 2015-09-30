#ifndef SHALLOW2D_H
#define SHALLOW2D_H

#include <cmath>
#include <array>

//ldoc on
/**
 * # Shallow water equations
 * 
 * ## Physics picture
 * 
 * The shallow water equations are a two-dimensional PDE system
 * that describes water waves that are very long compared to the
 * water depth.  It applies even in situations that you might not
 * think of as "shallow"; for example, tsunami waves are long enough
 * that they can be modeled using the shallow water equations even
 * when traveling over mile-deep parts of oceans.
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
 * I was inspired to use this system for our assignment by reading the
 * chapter on [shallow water simulation in MATLAB][exm] from Cleve
 * Moler's books on "Experiments in MATLAB"; there is also a very readable
 * [Wikipedia article][wiki] on the shallow water equations.
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
 * [exm]: https://www.mathworks.com/moler/exm/chapters/water.pdf
 * [wiki]: https://en.wikipedia.org/wiki/Shallow_water_equations
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

struct Shallow2D {

    // Type parameters for solver
    typedef float real;
    typedef std::array<real,3> vec;

    // Gravitational force (compile time constant)
    static constexpr real g = 9.8;

    // Compute shallow water fluxes F(U), G(U)
    static void flux(vec& FU, vec& GU, const vec& U) {
        real h = U[0], hu = U[1], hv = U[2];

        FU[0] = hu;
        FU[1] = hu*hu/h + (0.5*g)*h*h;
        FU[2] = hu*hv/h;

        GU[0] = hv;
        GU[1] = hu*hv/h;
        GU[2] = hv*hv/h + (0.5*g)*h*h;
    }

    // Compute shallow water wave speed
    static void wave_speed(real& cx, real& cy, const vec& U) {
        using namespace std;
        real h = U[0], hu = U[1], hv = U[2];
        real root_gh = sqrt(g * h);  // NB: Don't let h go negative!
        cx = abs(hu/h) + root_gh;
        cy = abs(hv/h) + root_gh;
    }
};

//ldoc off
#endif /* SHALLOW2D_H */
