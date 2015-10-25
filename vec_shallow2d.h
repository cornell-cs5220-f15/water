#ifndef VEC_SHALLOW2D_H
#define VEC_SHALLOW2D_H

#include <cmath>


struct Shallow2D {

    // Type parameters for solver
    typedef float real;

    // Gravitational force (compile time constant)
    static constexpr real g = 9.8;

    // Compute shallow water fluxes F(U), G(U)
    static void flux(real* FU, real* GU, const real* U) {
        real h = U[0], hu = U[1], hv = U[2];

        FU[0] = hu;
        FU[1] = hu*hu/h + (0.5*g)*h*h;
        FU[2] = hu*hv/h;

        GU[0] = hv;
        GU[1] = hu*hv/h;
        GU[2] = hv*hv/h + (0.5*g)*h*h;
    }

    // Compute shallow water wave speed
    static void wave_speed(real& cx, real& cy, const real* U) {
        using namespace std;
        real h = U[0], hu = U[1], hv = U[2];
        real root_gh = sqrt(g * h);  // NB: Don't let h go negative!
        cx = abs(hu/h) + root_gh;
        cy = abs(hv/h) + root_gh;
    }
};

#endif /* VEC_SHALLOW2D_H */
