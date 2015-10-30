#ifndef VEC_MINMOD_H
#define VEC_MINMOD_H

#include <cmath>


template <class real>
struct MinMod {
    static constexpr real theta = 2.0;

    // Branch-free computation of minmod of two numbers
    static real xmin(real a, real b) {
        using namespace std;
        return ((copysign((real) 0.5, a) +
                 copysign((real) 0.5, b)) *
                min( abs(a), abs(b) ));
    }

    // Limited combined slope estimate
    static real limdiff(real um, real u0, real up) {
        real du1 = u0-um;         // Difference to left
        real du2 = up-u0;         // Difference to right
        real duc = 0.5*(du1+du2); // Centered difference
        return xmin( theta*xmin(du1, du2), duc );
    }
};

#endif /* MINMOD_H */
