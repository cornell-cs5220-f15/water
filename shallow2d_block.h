#ifndef SHALLOW2D_BLOCK_H
#define SHALLOW2D_BLOCK_H

#include <array>
#include <cmath>

struct Shallow2DBlock {
    // Type parameters for solver
    typedef float real;
    typedef std::array<real,3> vec;

    // Number of fields in U vector
    static constexpr int num_fields = 3;

    // Gravitational force (compile time constant)
    static constexpr real g = 9.8;

    // Compute shallow water fluxes F(U), G(U)
    static void flux(real* restrict FU0, real* restrict FU1, real* restrict FU2,
                     real* restrict GU0, real* restrict GU1, real* restrict GU2,
                     const real h, const real hu, const real hv) {
        *FU0 = hu;
        *FU1 = hu*hu/h + (0.5f*g)*h*h;
        *FU2 = hu*hv/h;

        *GU0 = hv;
        *GU1 = hu*hv/h;
        *GU2 = hv*hv/h + (0.5f*g)*h*h;
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

#endif /* SHALLOW2D_BLOCK_H */
