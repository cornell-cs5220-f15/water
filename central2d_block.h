#ifndef CENTRAL2D_BLOCK_H
#define CENTRAL2D_BLOCK_H

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

template <class Physics, class Limiter>
class Central2DBlock {
public:
    typedef typename Physics::real real;
    typedef typename Physics::vec  vec;
    static constexpr int num_fields = Physics::num_fields;

    Central2DBlock(real w, real h,     // Domain width / height
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

    Central2DBlock(const Central2DBlock&) = delete;

    ~Central2DBlock() {
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
    static constexpr int nghost     = 3;  // Number of ghost cells
    static constexpr int block_size = 64; // The side length of one block

    const int nx, ny;           // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;   // Total cells in x/y (including ghost)
    const real dx, dy;          // Cell size in x/y
    const real cfl;             // Allowed CFL number

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
    static int offset(int nx_all, int ny_all, int k, int ix, int iy) {
        return (((k * ny_all) + iy) * nx_all) + ix;
    }

    int offset(int k, int ix, int iy) const {
        return (((k * ny_all) + iy) * nx_all) + ix;
    }

    // Wrapped accessor (periodic BC)
    static int ioffset(int nx_all, int ny_all, int nx, int ny, int nghost,
                       int k, int ix, int iy) {
        return offset(nx_all, ny_all, k,
                      (ix+nx-nghost) % nx + nghost,
                      (iy+ny-nghost) % ny + nghost);
    }

    int ioffset(int k, int ix, int iy) {
        return offset(k,
                      (ix+nx-nghost) % nx + nghost,
                      (iy+ny-nghost) % ny + nghost);
    }

    real& get(real *xs, int nx_all, int ny_all, int k, int ix, int iy) {
        return xs[offset(nx_all, ny_all, k, ix, iy)];
    }

    real& u (int k, int ix, int iy) { return get(u_,  nx_all, ny_all, k, ix, iy); }
    real& v (int k, int ix, int iy) { return get(v_,  nx_all, ny_all, k, ix, iy); }
    real& f (int k, int ix, int iy) { return get(f_,  nx_all, ny_all, k, ix, iy); }
    real& g (int k, int ix, int iy) { return get(g_,  nx_all, ny_all, k, ix, iy); }
    real& ux(int k, int ix, int iy) { return get(ux_, nx_all, ny_all, k, ix, iy); }
    real& uy(int k, int ix, int iy) { return get(uy_, nx_all, ny_all, k, ix, iy); }
    real& fx(int k, int ix, int iy) { return get(fx_, nx_all, ny_all, k, ix, iy); }
    real& gy(int k, int ix, int iy) { return get(gy_, nx_all, ny_all, k, ix, iy); }

    real& uwrap(int k, int ix, int iy) {
        return u_[ioffset(nx_all, ny_all, nx, ny, nghost, k, ix, iy)];
    }

    // Stages of the main algorithm
    void run_block(const int io, const real dt, const int bx, const int by);

    void apply_periodic();

    void compute_max_speed(real& cx, real& cy);
    void flux(real* restrict f0, real* restrict f1, real* restrict f2,
              real* restrict g0, real* restrict g1, real* restrict g2,
              const real* restrict u, const real* restrict hu, const real* restrict hv,
              const int len);
    void flux();

    void limited_derivsx(real* restrict dx,
                         const real* restrict x,
                         const int nx_all,
                         const int ny_all);
    void limited_derivsy(real* restrict dy,
                         const real* restrict y,
                         const int nx_all,
                         const int ny_all);
    void limited_derivs();

    void predictor_flux(real* restrict f0, real* restrict f1, real* restrict f2,
                        real* restrict g0, real* restrict g1, real* restrict g2,
                        const real* restrict fx0,
                        const real* restrict fx1,
                        const real* restrict fx2,
                        const real* restrict gy0,
                        const real* restrict gy1,
                        const real* restrict gy2,
                        const real* restrict u0,
                        const real* restrict u1,
                        const real* restrict u2,
                        const real dtcdx2, const real dtcdy2,
                        const int len);
    void corrector(real* restrict v,
                   const real* restrict u,
                   const real* restrict ux,
                   const real* restrict uy,
                   const real* restrict f,
                   const real* restrict g,
                   const real dtcdx2, const real dtcdy2,
                   const int nx,      const int ny,
                   const int nx_all,  const int ny_all,
                   const int bghost,  const int io);
    void compute_step(int io, real dt);
};

// Note that this function is not vectorized, but since `init` is only called
// once, we don't care about optimizing it.
template <class Physics, class Limiter>
template <typename F>
void Central2DBlock<Physics, Limiter>::init(F f) {
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

// VEC
// TODO(mwhittaker): convert ny_all to compile time constant.
template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::apply_periodic() {
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
void Central2DBlock<Physics, Limiter>::compute_max_speed(real& cx, real& cy) {
    using namespace std;
    real _cx = 1.0e-15;
    real _cy = 1.0e-15;

    for (int iy = nghost; iy < ny + nghost; ++iy) {
        for (int ix = nghost; ix < nx + nghost; ++ix) {
            real cell_cx, cell_cy;
            Physics::wave_speed(cell_cx, cell_cy,
                                u(0, ix,iy), u(1, ix, iy), u(2, ix, iy));
            _cx = max(_cx, cell_cx);
            _cy = max(_cy, cell_cy);
        }
    }

    cx = _cx;
    cy = _cy;
}

template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::flux(
        real* restrict f0, real* restrict f1, real* restrict f2,
        real* restrict g0, real* restrict g1, real* restrict g2,
        const real* restrict u, const real* restrict hu, const real* restrict hv,
        const int len) {
    for (int i = 0; i < len; ++i) {
        real _h  = u[i];
        real _hu = hu[i];
        real _hv = hv[i];

        f0[i] = _hu;
        f1[i] = _hu*_hu/_h + (0.5f*Physics::g)*_h*_h;
        f2[i] = _hu*_hv/_h;

        g0[i] = _hv;
        g1[i] = _hu*_hv/_h;
        g1[i] = _hv*_hv/_h + (0.5f*Physics::g)*_h*_h;
    }
}

template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::flux() {
    // KINDA VEC
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
}


template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::limited_derivsx(real* restrict _dx,
                                                      const real* restrict _x,
                                                      const int nx_all,
                                                      const int ny_all) {
    for (int x = 1; x < nx_all - 1; ++x) {
        for (int y = 1; y < ny_all - 1; ++y) {
            _dx[offset(nx_all, ny_all, 0, x, y)] = Limiter::limdiff(
                    _x[offset(nx_all, ny_all, 0, x-1, y)],
                    _x[offset(nx_all, ny_all, 0, x,   y)],
                    _x[offset(nx_all, ny_all, 0, x+1, y)]);
        }
    }
}

template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::limited_derivsy(real* restrict _dy,
                                                      const real* restrict _y,
                                                      const int nx_all,
                                                      const int ny_all) {
    for (int x = 1; x < nx_all - 1; ++x) {
        for (int y = 1; y < ny_all - 1; ++y) {
            _dy[offset(nx_all, ny_all, 0, x, y)] = Limiter::limdiff(
                    _y[offset(nx_all, ny_all, 0, x, y-1)],
                    _y[offset(nx_all, ny_all, 0, x, y)],
                    _y[offset(nx_all, ny_all, 0, x, y+1)]);
        }
    }
}

template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::limited_derivs() {
    using L = Limiter;
    for (int k = 0; k < num_fields; ++k) {
        for (int iy = 1; iy < ny_all-1; ++iy) {
            for (int ix = 1; ix < nx_all-1; ++ix) {
                // x derivs
                ux(k, ix, iy) = L::limdiff(u(k, ix-1,iy), u(k, ix,iy), u(k, ix+1,iy));
                fx(k, ix, iy) = L::limdiff(f(k, ix-1,iy), f(k, ix,iy), f(k, ix+1,iy));

                // y derivs
                uy(k, ix, iy) = L::limdiff(u(k, ix,iy-1), u(k, ix,iy), u(k, ix,iy+1));
                gy(k, ix, iy) = L::limdiff(g(k, ix,iy-1), g(k, ix,iy), g(k, ix,iy+1));
            }
        }
    }
}


template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::predictor_flux(
        real* restrict f0, real* restrict f1, real* restrict f2,
        real* restrict g0, real* restrict g1, real* restrict g2,
        const real* restrict fx0, const real* restrict fx1, const real* restrict fx2,
        const real* restrict gy0, const real* restrict gy1, const real* restrict gy2,
        const real* restrict u0,  const real* restrict u1,  const real* restrict u2,
        const real dtcdx2, const real dtcdy2, const int len) {
    for (int i = 0; i < len; ++i) {
        real h  = u0[i];
        real hu = u1[i];
        real hv = u2[i];

        h  -= dtcdx2 * fx0[i];
        h  -= dtcdy2 * gy0[i];
        hu -= dtcdx2 * fx1[i];
        hu -= dtcdy2 * gy1[i];
        hv -= dtcdx2 * fx2[i];
        hv -= dtcdy2 * gy2[i];

        Physics::flux(&f0[i], &f1[i], &f2[i], &g0[i], &g1[i], &g2[i], h, hu, hv);
    }
}

template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::corrector(
        real* restrict v,
        const real* restrict u,
        const real* restrict ux,
        const real* restrict uy,
        const real* restrict f,
        const real* restrict g,
        const real dtcdx2, const real dtcdy2,
        const int nx,      const int ny,
        const int nx_all,  const int ny_all,
        const int bghost,  const int io) {

    for (int iy = bghost-io; iy < ny+bghost-io; ++iy) {
        for (int ix = bghost-io; ix < nx+bghost-io; ++ix) {
            v[offset(nx_all, ny_all, 0, ix,iy)] =
                0.2500 * ( u[offset(nx_all, ny_all, 0, ix,   iy  )] +
                           u[offset(nx_all, ny_all, 0, ix+1, iy  )] +
                           u[offset(nx_all, ny_all, 0, ix,   iy+1)] +
                           u[offset(nx_all, ny_all, 0, ix+1, iy+1)] ) -

                0.0625 * ( ux[offset(nx_all, ny_all, 0, ix+1, iy  )] -
                           ux[offset(nx_all, ny_all, 0, ix,   iy  )] +
                           ux[offset(nx_all, ny_all, 0, ix+1, iy+1)] -
                           ux[offset(nx_all, ny_all, 0, ix,   iy+1)] +
                           uy[offset(nx_all, ny_all, 0, ix,   iy+1)] -
                           uy[offset(nx_all, ny_all, 0, ix,   iy  )] +
                           uy[offset(nx_all, ny_all, 0, ix+1, iy+1)] -
                           uy[offset(nx_all, ny_all, 0, ix+1, iy  )] ) -

                dtcdx2 * ( f[offset(nx_all, ny_all, 0,  ix+1, iy  )] -
                           f[offset(nx_all, ny_all, 0,  ix,   iy  )] +
                           f[offset(nx_all, ny_all, 0,  ix+1, iy+1)] -
                           f[offset(nx_all, ny_all, 0,  ix,   iy+1)] ) -

                dtcdy2 * ( g[offset(nx_all, ny_all, 0,  ix,   iy+1)] -
                           g[offset(nx_all, ny_all, 0,  ix,   iy  )] +
                           g[offset(nx_all, ny_all, 0,  ix+1, iy+1)] -
                           g[offset(nx_all, ny_all, 0,  ix+1, iy  )] );
        }
    }
}

template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::compute_step(int io, real dt)
{
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;

    // Predictor (flux values of f and g at half step)
    for (int iy = 1; iy < ny_all-1; ++iy)
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

    // Corrector (finish the step)
    for (int iy = nghost-io; iy < ny+nghost-io; ++iy)
        for (int ix = nghost-io; ix < nx+nghost-io; ++ix) {
            for (int k = 0; k < num_fields; ++k) {
                v(k, ix,iy) =
                    0.2500 * ( u(k, ix,  iy) + u(k, ix+1,iy  ) +
                               u(k, ix,iy+1) + u(k, ix+1,iy+1) ) -
                    0.0625 * ( ux(k, ix+1,iy  ) - ux(k, ix,iy  ) +
                               ux(k, ix+1,iy+1) - ux(k, ix,iy+1) +
                               uy(k, ix,  iy+1) - uy(k, ix,  iy) +
                               uy(k, ix+1,iy+1) - uy(k, ix+1,iy) ) -
                    dtcdx2 * ( f(k, ix+1,iy  ) - f(k, ix,iy  ) +
                               f(k, ix+1,iy+1) - f(k, ix,iy+1) ) -
                    dtcdy2 * ( g(k, ix,  iy+1) - g(k, ix,  iy) +
                               g(k, ix+1,iy+1) - g(k, ix+1,iy) );
            }
        }

    // Copy from v storage back to main grid
    for (int k = 0; k < num_fields; ++k) {
        for (int j = nghost; j < ny+nghost; ++j){
            for (int i = nghost; i < nx+nghost; ++i){
                u(k, i, j) = v(k, i-io, j-io);
            }
        }
    }
}

template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::run_block(const int io,
                                                 const real dt,
                                                 const int bx,
                                                 const int by) {
    //
    //    +---+---+---+---+---+---+---+---+
    //  6 |48#|49#|50#|51#|52#|53#|54#|55#|         +---+---+---+---+
    //    +---+---+---+---+---+---+---+---+       5 |###|###|###|###|
    //  5 |40#|41c|42c|43c|44c|45c|46d|47#|         +---+---+---+---+
    //    +---+---+---+---+---+---+---+---+       4 |###|38b|39b|###|
    //  4 |32#|33a|34a|35a|36a|37b|38b|39#|         +---+---+---+---+
    //    +---+---+---+---+---+---+---+---+       3 |###|30b|31b|###|
    //  3 |24#|25a|26a|27a|28a|29b|30b|31#|  ==>    +---+---+---+---+
    //    +---+---+---+---+---+---+---+---+       2 |###|21b|22b|###|
    //  2 |16#|17a|18a|19a|20a|21b|22b|23#|         +---+---+---+---+
    //    +---+---+---+---+---+---+---+---+       1 |###|13b|14b|###|
    //  1 ||8#| 9a|10a|11a|12a|13b|14b|15#|         +---+---+---+---+
    //    +---+---+---+---+---+---+---+---+       0 |###|###|###|###|
    //  0 |#0#|#1#|#2#|#3#|#4#|#5#|#6#|#7#|         +---+---+---+---+
    //    +---+---+---+---+---+---+---+---+           0   1   2   3
    //      0   1   2   3   4   5   6   7
    //
    //                             nghost = 1
    //                           block_size = 4
    //                           nx = 6, ny = 5
    //                       nx_all = 8, ny_all = 7
    //                           BX = 2, BY = 2

    // initialize sizes and dimensions
    const int bghosts = 3;
    const int ix = nghost + (bx * block_size);
    const int iy = nghost + (by * block_size);
    const int width  = (ix+block_size > nx+nghost) ? nx+nghost-ix : block_size;
    const int height = (iy+block_size > ny+nghost) ? ny+nghost-iy : block_size;
    const int width_all  = width  + 2*bghosts;
    const int height_all = height + 2*bghosts;
    const int size_all   = width_all * height_all;

    // allocate blocks
    real *_u  = (real *)malloc(size_all * num_fields * sizeof(real));
    real *_f  = (real *)malloc(size_all * num_fields * sizeof(real));
    real *_g  = (real *)malloc(size_all * num_fields * sizeof(real));
    real *_ux = (real *)malloc(size_all * num_fields * sizeof(real));
    real *_uy = (real *)malloc(size_all * num_fields * sizeof(real));
    real *_fx = (real *)malloc(size_all * num_fields * sizeof(real));
    real *_gy = (real *)malloc(size_all * num_fields * sizeof(real));
    real *_v  = (real *)malloc(size_all * num_fields * sizeof(real));

    // convenient aliases
    real *u0  = &_u [offset(width_all, height_all, 0, 0, 0)];
    real *u1  = &_u [offset(width_all, height_all, 1, 0, 0)];
    real *u2  = &_u [offset(width_all, height_all, 2, 0, 0)];
    real *f0  = &_f [offset(width_all, height_all, 0, 0, 0)];
    real *f1  = &_f [offset(width_all, height_all, 1, 0, 0)];
    real *f2  = &_f [offset(width_all, height_all, 2, 0, 0)];
    real *g0  = &_g [offset(width_all, height_all, 0, 0, 0)];
    real *g1  = &_g [offset(width_all, height_all, 1, 0, 0)];
    real *g2  = &_g [offset(width_all, height_all, 2, 0, 0)];
    real *ux0 = &_ux[offset(width_all, height_all, 0, 0, 0)];
    real *ux1 = &_ux[offset(width_all, height_all, 1, 0, 0)];
    real *ux2 = &_ux[offset(width_all, height_all, 2, 0, 0)];
    real *uy0 = &_uy[offset(width_all, height_all, 0, 0, 0)];
    real *uy1 = &_uy[offset(width_all, height_all, 1, 0, 0)];
    real *uy2 = &_uy[offset(width_all, height_all, 2, 0, 0)];
    real *fx0 = &_fx[offset(width_all, height_all, 0, 0, 0)];
    real *fx1 = &_fx[offset(width_all, height_all, 1, 0, 0)];
    real *fx2 = &_fx[offset(width_all, height_all, 2, 0, 0)];
    real *gy0 = &_gy[offset(width_all, height_all, 0, 0, 0)];
    real *gy1 = &_gy[offset(width_all, height_all, 1, 0, 0)];
    real *gy2 = &_gy[offset(width_all, height_all, 2, 0, 0)];
    real *v0  = &_v [offset(width_all, height_all, 0, 0, 0)];
    real *v1  = &_v [offset(width_all, height_all, 1, 0, 0)];
    real *v2  = &_v [offset(width_all, height_all, 2, 0, 0)];

    // copy from underlying u into our block u
    for (int k = 0; k < num_fields; ++k) {
        for (int x = 0; x < width_all; ++x) {
            for (int y = 0; y < height_all; ++y) {
                get(_u, width_all, height_all, k, x, y) = u(k, ix-1+x, iy-1+y);
            }
        }
    }

    // flux
    flux(f0, f1, f2, g0, g1, g2, u0, u1, u2, width_all * height_all);

    // limited_derivs
    limited_derivsx(ux0, u0, width_all, height_all);
    limited_derivsx(ux1, u1, width_all, height_all);
    limited_derivsx(ux2, u2, width_all, height_all);
    limited_derivsx(fx0, f0, width_all, height_all);
    limited_derivsx(fx1, f1, width_all, height_all);
    limited_derivsx(fx2, f2, width_all, height_all);

    limited_derivsy(uy0, u0, width_all, height_all);
    limited_derivsy(uy1, u1, width_all, height_all);
    limited_derivsy(uy2, u2, width_all, height_all);
    limited_derivsy(gy0, g0, width_all, height_all);
    limited_derivsy(gy1, g1, width_all, height_all);
    limited_derivsy(gy2, g2, width_all, height_all);

    // compute_step
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;
    predictor_flux(f0, f1, f2, g0, g1, g2,
                   fx0, fx1, fx2, gy0, gy1, gy2, u0, u1, u2,
                   dtcdx2, dtcdy2, width_all * height_all);

    free(_u);
    free(_f);
    free(_g);
    free(_ux);
    free(_uy);
    free(_fx);
    free(_gy);
    free(_v);
}

template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::run(real tfinal)
{
    const int BX = nx / block_size + (nx % block_size != 0 ? 1 : 0);
    const int BY = ny / block_size + (ny % block_size != 0 ? 1 : 0);

    bool done = false;
    real t = 0;
    while (!done) {
        real dt;
        for (int io = 0; io < 2; ++io) {
            real cx, cy;
            compute_max_speed(cx, cy);
            if (io == 0) {
                dt = cfl / std::max(cx/dx, cy/dy);
                if (t + 2*dt >= tfinal) {
                    dt = (tfinal-t)/2;
                    done = true;
                }
            }

            for (int by = 0; by < BY; ++by) {
                for (int bx = 0; bx < BX; ++bx) {
                    run_block(io, dt, bx, by);
                }
            }

            t += dt;
        }
    }
}

// Note that this function is not vectorized, but we don't check solutions when
// timing, so we don't have to optimize it.
template <class Physics, class Limiter>
void Central2DBlock<Physics, Limiter>::solution_check()
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

#endif /* CENTRAL2D_BLOCK_H*/
