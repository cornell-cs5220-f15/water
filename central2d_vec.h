#ifndef CENTRAL2D_VEC_H
#define CENTRAL2D_VEC_H

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

template <class Physics, class Limiter>
class Central2DVec {
public:
    typedef typename Physics::real real;
    typedef typename Physics::vec  vec;
    static constexpr int num_fields = Physics::num_fields;

    Central2DVec(real w, real h,     // Domain width / height
                 int nx, int ny,     // Number of cells in x/y (without ghosts)
                 real cfl = 0.45) :  // Max allowed CFL number
        nx(nx), ny(ny),
        nx_all(nx + 2*nghost),
        ny_all(ny + 2*nghost),
        dx(w/nx), dy(h/ny),
        cfl(cfl),
        u_ ((real *)malloc(sizeof(real) * num_fields * nx_all * ny_all)),
        f_ ((real *)malloc(sizeof(real) * num_fields * nx_all * ny_all)),
        g_ ((real *)malloc(sizeof(real) * num_fields * nx_all * ny_all)),
        ux_((real *)malloc(sizeof(real) * num_fields * nx_all * ny_all)),
        uy_((real *)malloc(sizeof(real) * num_fields * nx_all * ny_all)),
        fx_((real *)malloc(sizeof(real) * num_fields * nx_all * ny_all)),
        gy_((real *)malloc(sizeof(real) * num_fields * nx_all * ny_all)),
        v_ ((real *)malloc(sizeof(real) * num_fields * nx_all * ny_all)) {}

    Central2DVec(const Central2DVec&) = delete;

    ~Central2DVec() {
        free(u_);
        free(f_);
        free(g_);
        free(ux_);
        free(uy_);
        free(fx_);
        free(gy_);
        free(v_);
    }

    // Advance from time 0 to time tfinal
    void run(real tfinal);

    // Call f(Uxy, x, y) at each cell center to set initial conditions
    template <typename F>
    void init(F f);

    // Diagnostics
    void solution_check();

    // Array size accessors
    int xsize() const { return nx; }
    int ysize() const { return ny; }

    // Read elements of simulation state
    // Note that this function is not vectorized, but it is only called by
    // `meshio` and is not used when timing.
    const vec operator()(int i, int j) const {
        vec v;
        for (int k = 0; k < num_fields; ++k) {
            v[k] = u_[offset(k, i+nghost,j+nghost)];
        }
        return v;
    }

private:
    static constexpr int nghost = 3;   // Number of ghost cells

    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number

    // All vectors are stored as a flattened representation of a
    // two-dimensional array of arrays. For example, consider a 2x4 grid where
    // each element of the grid contains a vector of length three.
    //
    //                        +---+---+---+---+
    //                      1 | d | f | g | h |
    //                        +---+---+---+---+
    //                      0 | a | b | c | d |
    //                        +---+---+---+---+
    //                          0   1   2   3
    //
    // This grid is stored as follows:
    //
    //     +---+---+---+---+---+---+---+---+---+   +---+---+   +---+
    //     |a_0|b_0|c_0|d_0|e_0|f_0|g_0|h_0|a_1|...|h_1|a_2|...|h_2|
    //     +---+---+---+---+---+---+---+---+---+   +---+---+   +---+
    //       0   1   2   3   4   5   6   7   8      15  16      23
    //
    // This representation is taken from Prof. Bindel's implementation:
    // https://github.com/dbindel/water
    real *u_;  // Solution values
    real *f_;  // Fluxes in x
    real *g_;  // Fluxes in y
    real *ux_; // x differences of u
    real *uy_; // y differences of u
    real *fx_; // x differences of f
    real *gy_; // y differences of g
    real *v_;  // Solution values at next step

    // Array accessor functions
    int offset(int k, int ix, int iy) const {
        return (((k * ny_all) + iy) * nx_all) + ix;
    }

    // Wrapped accessor (periodic BC)
    int ioffset(int k, int ix, int iy) {
        return offset(k,
                      (ix+nx-nghost) % nx + nghost,
                      (iy+ny-nghost) % ny + nghost);
    }

    real& u    (int k, int ix, int iy) { return u_ [offset (k, ix, iy)]; }
    real& v    (int k, int ix, int iy) { return v_ [offset (k, ix, iy)]; }
    real& f    (int k, int ix, int iy) { return f_ [offset (k, ix, iy)]; }
    real& g    (int k, int ix, int iy) { return g_ [offset (k, ix, iy)]; }
    real& ux   (int k, int ix, int iy) { return ux_[offset (k, ix, iy)]; }
    real& uy   (int k, int ix, int iy) { return uy_[offset (k, ix, iy)]; }
    real& fx   (int k, int ix, int iy) { return fx_[offset (k, ix, iy)]; }
    real& gy   (int k, int ix, int iy) { return gy_[offset (k, ix, iy)]; }
    real& uwrap(int k, int ix, int iy) { return u_ [ioffset(k, ix, iy)]; }

    // Stages of the main algorithm
    void apply_periodic();
    void compute_fg_speeds(real& cx, real& cy);
    void limited_derivs();
    void compute_step(int io, real dt);
};

// Note that this function is not vectorized, but since `init` is only called
// once, we don't care about optimizing it.
template <class Physics, class Limiter>
template <typename F>
void Central2DVec<Physics, Limiter>::init(F f) {
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            vec v;
            for (int k = 0; k < num_fields; ++k) {
                v[k] = u(k, nghost+ix, nghost+iy);
            }
            f(v, (ix+0.5)*dx, (iy+0.5)*dy);
            for (int k = 0; k < num_fields; ++k) {
                u(k, nghost+ix, nghost+iy) = v[k];
            }
        }
    }
}

template <class Physics, class Limiter>
void Central2DVec<Physics, Limiter>::apply_periodic() {
    // Copy data between right and left boundaries
    for (int iy = 0; iy < ny_all; ++iy) {
        for (int ix = 0; ix < nghost; ++ix) {
            #pragma ivdep
            for (int k = 0; k < num_fields; ++k) {
                u(k, ix,          iy) = uwrap(k, ix,          iy);
                u(k, nx+nghost+ix,iy) = uwrap(k, nx+nghost+ix,iy);
            }
        }
    }

    // Copy data between top and bottom boundaries
    for (int ix = 0; ix < nx_all; ++ix) {
        for (int iy = 0; iy < nghost; ++iy) {
            #pragma ivdep
            for (int k = 0; k < num_fields; ++k) {
                u(k, ix,          iy) = uwrap(k, ix,          iy);
                u(k, ix,ny+nghost+iy) = uwrap(k, ix,ny+nghost+iy);
            }
        }
    }
}

template <class Physics, class Limiter>
void Central2DVec<Physics, Limiter>::compute_fg_speeds(real& cx_, real& cy_)
{
    using namespace std;
    real cx = 1.0e-15;
    real cy = 1.0e-15;

    for (int iy = 0; iy < ny_all; ++iy) {
        #pragma ivdep
        for (int ix = 0; ix < nx_all; ++ix) {
            real& f0 = f(0, ix, iy);
            real& f1 = f(1, ix, iy);
            real& f2 = f(2, ix, iy);
            real& g0 = g(0, ix, iy);
            real& g1 = g(1, ix, iy);
            real& g2 = g(2, ix, iy);
            real& h  = u(0, ix, iy);
            real& hu = u(1, ix, iy);
            real& hv = u(2, ix, iy);

            Physics::flux(f0, f1, f2, g0, g1, g2, h, hu, hv);
        }
    }

    for (int iy = 0; iy < ny_all; ++iy) {
        for (int ix = 0; ix < nx_all; ++ix) {
            real cell_cx, cell_cy;
            Physics::wave_speed(cell_cx, cell_cy,
                                u(0, ix,iy), u(1, ix, iy), u(2, ix, iy));
            cx = max(cx, cell_cx);
            cy = max(cy, cell_cy);
        }
    }

    cx_ = cx;
    cy_ = cy;
}

template <class Physics, class Limiter>
void Central2DVec<Physics, Limiter>::limited_derivs() {
    using L = Limiter;

    for (int k = 0; k < num_fields; ++k) {
        for (int iy = 1; iy < ny_all-1; ++iy) {
            #pragma vector aligned
            #pragma ivdep
            for (int ix = 1; ix < nx_all-1; ++ix) {
                // x derivs
                ux(k, ix, iy) = L::limdiff(u(k, ix-1,iy), u(k, ix,iy), u(k, ix+1,iy));
                fx(k, ix, iy) = L::limdiff(f(k, ix-1,iy), f(k, ix,iy), f(k, ix+1,iy));
            }
        }

        for (int ix = 1; ix < nx_all-1; ++ix) {
            #pragma vector aligned
            #pragma ivdep
            for (int iy = 1; iy < ny_all-1; ++iy) {
                // y derivs
                uy(k, ix, iy) = L::limdiff(u(k, ix,iy-1), u(k, ix,iy), u(k, ix,iy+1));
                gy(k, ix, iy) = L::limdiff(g(k, ix,iy-1), g(k, ix,iy), g(k, ix,iy+1));
            }
        }
    }
}

template <class Physics, class Limiter>
void Central2DVec<Physics, Limiter>::compute_step(int io, real dt)
{
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;

    // Predictor (flux values of f and g at half step)
    for (int iy = 1; iy < ny_all-1; ++iy) {
        #pragma vector aligned
        #pragma ivdep
        for (int ix = 1; ix < nx_all-1; ++ix) {
            real h  = u(0, ix, iy);
            real hu = u(1, ix, iy);
            real hv = u(2, ix, iy);

            h  -= dtcdx2 * fx(0, ix, iy);
            h  -= dtcdy2 * gy(0, ix, iy);
            hu -= dtcdx2 * fx(1, ix, iy);
            hu -= dtcdy2 * gy(1, ix, iy);
            hv -= dtcdx2 * fx(2, ix, iy);
            hv -= dtcdy2 * gy(2, ix, iy);

            Physics::flux(f(0, ix,iy), f(1, ix, iy), f(2, ix, iy),
                          g(0, ix,iy), g(1, ix, iy), g(2, ix, iy),
                          h, hu, hv);
        }
    }

    // Corrector (finish the step)
    int _nx = nx;
    int _ny = ny;
    int _nghost = nghost;
    for (int k = 0; k < num_fields; ++k) {
        for (int iy = _nghost-io; iy < _ny+_nghost-io; ++iy) {
            for (int ix = _nghost-io; ix < _nx+_nghost-io; ++ix) {
                v(k, ix+io, iy+io) =
                    0.2500f * ( u(k, ix,  iy) + u(k, ix+1,iy  ) +
                                u(k, ix,iy+1) + u(k, ix+1,iy+1) ) -
                    0.0625f * ( ux(k, ix+1,iy  ) - ux(k, ix,iy  ) +
                                ux(k, ix+1,iy+1) - ux(k, ix,iy+1) +
                                uy(k, ix,  iy+1) - uy(k, ix,  iy) +
                                uy(k, ix+1,iy+1) - uy(k, ix+1,iy) ) -
                    dtcdx2 * (  f(k, ix+1,iy  ) - f(k, ix,iy  ) +
                                f(k, ix+1,iy+1) - f(k, ix,iy+1) ) -
                    dtcdy2 * (  g(k, ix,  iy+1) - g(k, ix,  iy) +
                                g(k, ix+1,iy+1) - g(k, ix+1,iy) );
            }
        }
    }

    std::swap(u_, v_);
}

template <class Physics, class Limiter>
void Central2DVec<Physics, Limiter>::run(real tfinal)
{
    bool done = false;
    real t = 0;
    while (!done) {
        real dt;
        for (int io = 0; io < 2; ++io) {
            real cx, cy;
            apply_periodic();
            compute_fg_speeds(cx, cy);
            limited_derivs();
            if (io == 0) {
                dt = cfl / std::max(cx/dx, cy/dy);
                if (t + 2*dt >= tfinal) {
                    dt = (tfinal-t)/2;
                    done = true;
                }
            }
            compute_step(io, dt);
            t += dt;
        }
    }
}

// Note that this function is not vectorized, but we don't check solutions when
// timing, so we don't have to optimize it.
template <class Physics, class Limiter>
void Central2DVec<Physics, Limiter>::solution_check()
{
    using namespace std;
    real h_sum = 0, hu_sum = 0, hv_sum = 0;
    real hmin = u(0, nghost, nghost);
    real hmax = hmin;
    for (int j = nghost; j < ny+nghost; ++j) {
        for (int i = nghost; i < nx+nghost; ++i) {
            real h = u(0, i, j);
            h_sum += h;
            hu_sum += u(1, i, j);
            hv_sum += u(2, i, j);
            hmax = max(h, hmax);
            hmin = min(h, hmin);
            assert(h > 0);
        }
    }
    real cell_area = dx*dy;
    h_sum *= cell_area;
    hu_sum *= cell_area;
    hv_sum *= cell_area;
    printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n",
           h_sum, hu_sum, hv_sum, hmin, hmax);
}

#endif /* CENTRAL2D_VEC_H*/
