#ifndef MINMOD_OPT_H
#define MINMOD_OPT_H

#include <cmath>

template <class real>
struct MinModOpt {
    static constexpr real theta   = 2.0f;
    static constexpr real quarter = 0.25f;

    // Branch-free computation of minmod of two numbers times 2s
    static real xmin2s(real s, real a, real b) {
        real sa = copysignf(s, a);
        real sb = copysignf(s, b);
        real abs_a = fabsf(a);
        real abs_b = fabsf(b);
        real min_abs = (abs_a < abs_b ? abs_a : abs_b);
        return (sa+sb) * min_abs;
    }


    // Limited combined slope estimate
    static real limdiff(real um, real u0, real up) {
        real du1 = u0-um; // Difference to left
        real du2 = up-u0; // Difference to right
        real duc = up-um; // Twice centered difference
        return xmin2s(quarter, xmin2s(theta, du1, du2), duc);
    }
};

#endif /* MINMOD_OPT_H */
