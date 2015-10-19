#ifndef SHALLOW2DVEC_H
#define SHALLOW2DVEC_H

#include <cmath>
#include <array>
#include <vector>
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
  typedef std::vector<real>::iterator iter;

  // Gravitational force (compile time constant)
  static constexpr real g = 9.8;

  // Compute shallow water fluxes F(U), G(U) for all points in the grid
  static void vflux(iter F,iter FU, iter FV , iter G,iter GU, iter GV,
		    iter h, iter hu, iter hv, int ncell)
  {
    std::copy(hu,hu+ncell,F);
    std::copy(hv,hv+ncell,G);
    for (int i = 0; i < ncell; ++i, ++h,++hu,++hv,++FU,++FV,++GU,++GV){
      real hi = (*h), hui = (*hu), hvi = (*hv);
      real inv_h = 1/hi;
      (*FU) = hui*hui*inv_h + (0.5f*g)*hi*hi;
      (*FV) = hui*hvi*inv_h;
      (*GU) = hui*hvi*inv_h;
      (*GV) = hvi*hvi*inv_h + (0.5f*g)*hi*hi;
    }
  }

  static void flux(iter FU, iter GU, const iter U,
		   int ncell, int field_stride)
  {
    vflux(FU,FU+field_stride,FU+2*field_stride,
	  GU,GU+field_stride,GU+2*field_stride,
	  U,U+field_stride,U+2*field_stride,ncell);
  }

  // Compute shallow water wave speed
  static void wavev_speed(real& cx, real& cy, iter h, iter hu,
			  iter hv, int ncell) {
    using namespace std;
    real cx2,cy2;
    for (int i = 0; i < ncell; ++i,++h,++hu,++hv){
        real hi = *h;
        real inv_hi=1/hi;
        real root_gh = sqrt(g * hi);  // NB: Don't let h go negative!
        cx2 = fabs((*hu) * inv_hi) + root_gh;
        cy2 = fabs((*hv) * inv_hi) + root_gh;
        if (cx < cx2) cx=cx2;
        if (cy < cy2) cy=cy2;
    }
  }

  static void wave_speed(real& cx, real& cy, const iter U,
			 int ncell, int field_stride)
  {
    wavev_speed(cx,cy,U,U+field_stride,U+2*field_stride,ncell);
  }
};

//ldoc off
#endif /* SHALLOW2D_H */
