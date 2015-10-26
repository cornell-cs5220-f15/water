#ifndef CENTRAL2D_BLOCK2_H
#define CENTRAL2D_BLOCK2_H

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "flat_array.h"

////////////////////////////////////////////////////////////////////////////////
// Block
////////////////////////////////////////////////////////////////////////////////
template <class Physics, class Limiter>
class Block {
public:
    typedef typename Physics::real real;
    static constexpr int num_fields = Physics::num_fields;

    Block(real *u, int nx, int ny, int nghost, real dx, real dy, int io, real dt) :
        nx_(nx),
        ny_(ny),
        nghost_(nghost),
        dx_(dx),
        dy_(dy),
        io_(io),
        dt_(dt),
        nx_all_(nx_ + 2*nghost_),
        ny_all_(ny_ + 2*nghost_),
        u_ (u),
        f_ (flat_array::make<real>(nx_all_, ny_all_, num_fields)),
        g_ (flat_array::make<real>(nx_all_, ny_all_, num_fields)),
        ux_(flat_array::make<real>(nx_all_, ny_all_, num_fields)),
        uy_(flat_array::make<real>(nx_all_, ny_all_, num_fields)),
        fx_(flat_array::make<real>(nx_all_, ny_all_, num_fields)),
        gy_(flat_array::make<real>(nx_all_, ny_all_, num_fields)),
        v_ (flat_array::make<real>(nx_all_, ny_all_, num_fields)) {}

    Block(const Block&) = delete;
    Block(Block&&)      = delete;

    ~Block() {
        free(f_);
        free(g_);
        free(ux_);
        free(uy_);
        free(fx_);
        free(gy_);
        free(v_);
    }

    void step();

private:
    const int nx_;
    const int ny_;
    const int nghost_;
    const real dx_;
    const real dy_;
    const int io_;
    const real dt_;
    const int nx_all_;
    const int ny_all_;
    real *u_;
    real *f_;
    real *g_;
    real *ux_;
    real *uy_;
    real *fx_;
    real *gy_;
    real *v_;

    real *at(real *xs, int x, int y) {
        return flat_array::at(xs, nx_all_, ny_all_, x, y);
    }

    real *field(real *xs, int k) {
        return flat_array::field(xs, nx_all_, ny_all_, k);
    }

    #define BLOCK_MAP(function) \
        function(u) \
        function(f) \
        function(g) \
        function(ux) \
        function(uy) \
        function(fx) \
        function(gy) \
        function(v)

    #define BLOCK_FIELDS(xs) \
        real *xs##0() { return field(xs##_, 0); } \
        real *xs##1() { return field(xs##_, 1); } \
        real *xs##2() { return field(xs##_, 2); }

    #define BLOCK_INDEXERS(xs) \
        real *xs(int k, int x, int y) { return at(field(xs##_, k), x, y); }

    BLOCK_MAP(BLOCK_FIELDS)
    BLOCK_MAP(BLOCK_INDEXERS)

private:
    void flux();
    void limited_derivs();
    void compute_step();
};

template <class Physics, class Limiter>
void Block<Physics, Limiter>::flux() {
    for (int y = 0; y < ny_all_; ++y) {
        for (int x = 0; x < nx_all_; ++x) {
            Physics::flux( f(0, x, y),  f(1, x, y),  f(2, x, y),
                           g(0, x, y),  g(1, x, y),  g(2, x, y),
                          *u(0, x, y), *u(1, x, y), *u(2, x, y));
        }
    }
}

template <class Physics, class Limiter>
void Block<Physics, Limiter>::limited_derivs() {
    using L = Limiter;
    for (int k = 0; k < num_fields; ++k) {
        for (int y = 1; y < ny_all_-1; ++y) {
            for (int x = 1; x < nx_all_-1; ++x) {
                // x derivs
                *ux(k, x, y) = L::limdiff(*u(k,x-1,y), *u(k,x,y), *u(k,x+1,y));
                *fx(k, x, y) = L::limdiff(*f(k,x-1,y), *f(k,x,y), *f(k,x+1,y));

                // y derivs
                *uy(k, x, y) = L::limdiff(*u(k,x,y-1), *u(k,x,y), *u(k,x,y+1));
                *gy(k, x, y) = L::limdiff(*g(k,x,y-1), *g(k,x,y), *g(k,x,y+1));
            }
        }
    }
}

template <class Physics, class Limiter>
void Block<Physics, Limiter>::compute_step() {
    real dtcdx2 = 0.5 * dt_ / dx_;
    real dtcdy2 = 0.5 * dt_ / dy_;

    // Predictor (flux values of f and g at half step)
    for (int y = 1; y < ny_all_-1; ++y) {
        for (int x = 1; x < nx_all_-1; ++x) {
            real h  = *u(0, x, y);
            real hu = *u(1, x, y);
            real hv = *u(2, x, y);

            h  -= dtcdx2 * *fx(0, x, y);
            h  -= dtcdy2 * *gy(0, x, y);
            hu -= dtcdx2 * *fx(1, x, y);
            hu -= dtcdy2 * *gy(1, x, y);
            hv -= dtcdx2 * *fx(2, x, y);
            hv -= dtcdy2 * *gy(2, x, y);

            Physics::flux(f(0, x,y), f(1, x, y), f(2, x, y),
                          g(0, x,y), g(1, x, y), g(2, x, y),
                          h, hu, hv);
        }
    }

    // Corrector (finish the step)
    for (int k = 0; k < num_fields; ++k) {
        for (int y = nghost_-io_; y < ny_+nghost_-io_; ++y)
            for (int x = nghost_-io_; x < nx_+nghost_-io_; ++x) {
                *v(k, x,y) =
                    0.2500 * ( *u(k, x,  y) + *u(k, x+1,y  ) +
                               *u(k, x,y+1) + *u(k, x+1,y+1) ) -
                    0.0625 * ( *ux(k, x+1,y  ) - *ux(k, x,y  ) +
                               *ux(k, x+1,y+1) - *ux(k, x,y+1) +
                               *uy(k, x,  y+1) - *uy(k, x,  y) +
                               *uy(k, x+1,y+1) - *uy(k, x+1,y) ) -
                    dtcdx2 * ( *f(k, x+1,y  ) - *f(k, x,y  ) +
                               *f(k, x+1,y+1) - *f(k, x,y+1) ) -
                    dtcdy2 * ( *g(k, x,  y+1) - *g(k, x,  y) +
                               *g(k, x+1,y+1) - *g(k, x+1,y) );
            }
        }

    // Copy from v storage back to main grid
    for (int k = 0; k < num_fields; ++k) {
        for (int y = nghost_; y < ny_+nghost_; ++y){
            for (int x = nghost_; x < nx_+nghost_; ++x){
                *u(k, x, y) = *v(k, x-io_, y-io_);
            }
        }
    }
}

template <class Physics, class Limiter>
void Block<Physics, Limiter>::step() {
    flux();
    limited_derivs();
    compute_step();
}

////////////////////////////////////////////////////////////////////////////////
// Central2D
////////////////////////////////////////////////////////////////////////////////
template <class Physics, class Limiter>
class Central2DBlock2 {
public:
    typedef typename Physics::real real;
    typedef typename Physics::vec  vec;
    static constexpr int num_fields = Physics::num_fields;

    Central2DBlock2(real w, real h,     // Domain width / height
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

    Central2DBlock2(const Central2DBlock2&) = delete;

    ~Central2DBlock2() {
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

    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number

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
    void run_block(const int io, const real dt, const int bx, const int by);
    void apply_periodic();
    void compute_max_speed(real& cx, real& cy);
};

// Note that this function is not vectorized, but since `init` is only called
// once, we don't care about optimizing it.
template <class Physics, class Limiter>
template <typename F>
void Central2DBlock2<Physics, Limiter>::init(F f) {
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
void Central2DBlock2<Physics, Limiter>::apply_periodic() {
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
void Central2DBlock2<Physics, Limiter>::compute_max_speed(real& cx, real& cy) {
    using namespace std;
    real _cx = 1.0e-15;
    real _cy = 1.0e-15;

    for (int iy = nghost; iy < ny + nghost; ++iy) {
        for (int ix = nghost; ix < nx + nghost; ++ix) {
            real cell_cx, cell_cy;
            Physics::wave_speed(cell_cx, cell_cy,
                                u(0, ix, iy), u(1, ix, iy), u(2, ix, iy));
            _cx = max(_cx, cell_cx);
            _cy = max(_cy, cell_cy);
        }
    }

    cx = _cx;
    cy = _cy;
}

template <class Physics, class Limiter>
void Central2DBlock2<Physics, Limiter>::run_block(const int io,
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

    // copy from u to _u
    real *_u  = flat_array::make<real>(width_all, height_all, num_fields);
    for (int k = 0; k < num_fields; ++k) {
        real *_uk = flat_array::field(_u, width_all, height_all, k);
        for (int x = 0; x < width_all; ++x) {
            for (int y = 0; y < height_all; ++y) {
                *flat_array::at(_uk, width_all, height_all, x, y) =
                    u(k, ix-bghosts+x, iy-bghosts+y);
            }
        }
    }

    // step block
    Block<Physics, Limiter> b(_u, width, height, bghosts, dx, dy, io, dt);
    b.step();

    // write back from _u to v
    for (int k = 0; k < num_fields; ++k) {
        real *_uk = flat_array::field(_u, width_all, height_all, k);
            for (int y = bghosts; y < height + bghosts; ++y) {
                for (int x = bghosts; x < width + bghosts; ++x) {
                v(k, ix-bghosts+x, iy-bghosts+y) =
                    *flat_array::at(_uk, width_all, height_all, x, y);
            }
        }
    }
    free(_u);
}


template <class Physics, class Limiter>
void Central2DBlock2<Physics, Limiter>::run(real tfinal)
{
    const int BX = nx / block_size + (nx % block_size != 0 ? 1 : 0);
    const int BY = ny / block_size + (ny % block_size != 0 ? 1 : 0);

    bool done = false;
    real t = 0;
    while (!done) {
        real dt;
        for (int io = 0; io < 2; ++io) {
            real cx, cy;
            apply_periodic();
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

            // Copy from v storage back to main grid
            for (int k = 0; k < num_fields; ++k) {
                for (int y = nghost; y < ny+nghost; ++y){
                    for (int x = nghost; x < nx+nghost; ++x){
                        u(k, x, y) = v(k, x, y);
                    }
                }
            }

            t += dt;
        }
    }
}

// Note that this function is not vectorized, but we don't check solutions when
// timing, so we don't have to optimize it.
template <class Physics, class Limiter>
void Central2DBlock2<Physics, Limiter>::solution_check()
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

#endif /* CENTRAL2D_BLOCK2_H*/
