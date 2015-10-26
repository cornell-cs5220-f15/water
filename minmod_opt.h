#ifndef MINMOD_OPT_H
#define MINMOD_OPT_H

#include <cmath>

template <class real>
struct MinModOpt {
    static constexpr real theta   = 2.0f;
    static constexpr real quarter = 0.25f;

    // Branch-free computation of minmod of two numbers times 2s
    static float xmin2s(float s, float a, float b) {
        float sa = copysignf(s, a);
        float sb = copysignf(s, b);
        float abs_a = fabsf(a);
        float abs_b = fabsf(b);
        float min_abs = (abs_a < abs_b ? abs_a : abs_b);
        return (sa+sb) * min_abs;
    }


    // Limited combined slope estimate
    static float limdiff(float um, float u0, float up) {
        float du1 = u0-um; // Difference to left
        float du2 = up-u0; // Difference to right
        float duc = up-um; // Twice centered difference
        return xmin2s(quarter, xmin2s(theta, du1, du2), duc);
    }
};

#endif /* MINMOD_OPT_H */
