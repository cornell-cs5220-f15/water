#ifndef SHALLOW2D_H
#define SHALLOW2D_H

#include <cmath>
#include <array>
#include <algorithm>

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

struct Shallow2D {

    // Type parameters for solver
    typedef float real;
    typedef std::vector<real> vec;

    // Gravitational force (compile time constant)
    static constexpr real g = 9.8;

    /*
     * The three components of the flux are represented in separate 
     * vectors to improve performance. Also, h, hu, and hv are stored
     * in separate vectors. Therefore, updates to flux can be done
     * in 9 distinct decoupled stages. 
     *
     * --------------------------------------------------------------------------
     * Vector component | hu operation | hv operation | h operation
     * --------------------------------------------------------------------------
     * FU[0]            | hu           | hu           | hu
     * FU[1]            | hu * hu      | hu * hu      | (hu * hu)/h + (0.5*g)*h*h
     * FU[2]            | hu           | hu * hv      | (hu * hv)/h
     * --------------------------------------------------------------------------
     *
     * --------------------------------------------------------------------------
     * Vector component | hv operation | hu operation | h operation
     * --------------------------------------------------------------------------
     * GU[0]            | hv           | hv           | hv
     * GU[1]            | hv           | hv * hu      | (hv * hu)/h 
     * GU[2]            | hv * hv      | hv * hv      | (hv * hv)/h + (0.5*g)*h*h
     * --------------------------------------------------------------------------
     */
    static void flux(
            vec& f0, vec& f1, vec& f2, 
            vec& g0, vec& g1, vec& g2,
            vec& u_h, vec& u_hu, vec& u_hv, 
            int x_begin, int x_end, int y_begin, int y_end, int nx) {
    
        f0 = u_hu;
        g0 = u_hv;

        // #pragma ivdep
        for(int i=0; i<n; i++) {
            real h = u_h[i];
            real hu = u_hu[i];
            real hv = u_hv[i];
            f1[i] = (hu * hu)/h + 0.5 * g * h * h;
            f2[i] = (hu * hv)/h;
            g1[i] = (hu * hv)/h;
            g2[i] = (hv * hv)/h + 0.5 * g * h * h;
        }
    }

    static void wave_speed(real &cx, real &cy, 
            const vec& u_h, const vec& u_hu, const vec& u_hv, int n) {

        for(int i=0; i<n; i++) {
            real h = u_h[i];
            real hu = u_hu[i];
            real hv = u_hv[i];
            real root_gh = sqrt(g * h);  // NB: Don't let h go negative!
            real cx_c = std::fabs(hu/h) + root_gh;
            real cy_c = std::fabs(hv/h) + root_gh;
            if (cx_c > cx) 
                cx = cx_c;
            if (cy_c > cy)
                cy = cy_c;
        }
    }
};

//ldoc off
#endif /* SHALLOW2D_H */
