#ifndef SHALLOW2D_H
#define SHALLOW2D_H

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
 * ## Interface
 *
 * To provide a general interface, we make the flux and speed functions
 * take arrays that consist of the `h`, `hu`, and `hv` components
 * in sequential arrays separated by `field_stride`: for example,
 * the start of the height field data is at `U`, the start of
 * the $x$ momentum is at `U+field_stride`, and the start of the
 * $y$ momentum is at `U+2*field_stride`.
 */

void shallow2d_flux(float* FU, float* GU, const float* U,
                    int ncell, int field_stride);
void shallow2d_speed(float* cxy, const float* U,
                     int ncell, int field_stride);

//ldoc off
#endif /* SHALLOW2D_H */
