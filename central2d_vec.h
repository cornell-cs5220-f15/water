#ifndef CENTRAL2D_VEC_H
#define CENTRAL2D_VEC_H

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

//ldoc on
/**
 * # Jiang-Tadmor central difference scheme
 *
 * [Jiang and Tadmor][jt] proposed a high-resolution finite difference
 * scheme for solving hyperbolic PDE systems in two space dimensions.
 * The method is particularly attractive because, unlike many other
 * methods in this space, it does not require that we write any
 * solvers for problems with special initial data (so-called Riemann
 * problems), nor even that we compute Jacobians of the flux
 * functions.
 *
 * While this code is based loosely on the Fortran code at the end of
 * Jiang and Tadmor's paper, we've written the current code to be
 * physics-agnostic (rather than hardwiring it to the shallow water
 * equations -- or the Euler equations in the Jiang-Tadmor paper).
 * If you're interested in the Euler equations, feel free to add your
 * own physics class to support them!
 *
 * [jt]: http://www.cscamm.umd.edu/tadmor/pub/central-schemes/Jiang-Tadmor.SISSC-98.pdf
 *
 * ## Staggered grids
 *
 * The Jiang-Tadmor scheme works by alternating between a main grid
 * and a staggered grid offset by half a step in each direction.
 * Understanding this is important, particularly if you want to apply
 * a domain decomposition method and batch time steps between
 * synchronization barriers in your parallel code!
 *
 * In even-numbered steps, the entry `u(i,j)` in the array of solution
 * values represents the average value of a cell centered at a point
 * $(x_i,y_j)$.  At the following odd-numbered step, the same entry
 * represents values for a cell centered at $(x_i + \Delta x/2, y_j +
 * \Delta y/2)$.  However, whenever we run a simulation, we always take
 * an even number of steps, so that outside the solver we can just think
 * about values on the main grid.  If `uold` and `unew` represent the
 * information at two successive *even* time steps (i.e. they represent
 * data on the same grid), then `unew(i,j)` depends indirectly on
 * `u(p,q)` for $i-3 \leq p \leq i+3$ and $j-3 \leq q \leq j+3$.
 *
 * We currently manage this implicitly: the arrays at even time steps
 * represent cell values on the main grid, and arrays at odd steps
 * represent cell values on the staggered grid.  Our main `run`
 * function always takes an even number of time steps to ensure we end
 * up on the primary grid.
 *
 * ## Interface
 *
 * We want a clean separation between the physics, the solver,
 * and the auxiliary limiter methods used by the solver.  At the same
 * time, we don't want to pay the overhead (mostly in terms of lost
 * optimization opportunities) for calling across an abstraction
 * barrier in the inner loops of our solver.  We can get around this
 * in C++ by providing the solver with *template arguments*, resolved
 * at compile time, that describe separate classes to implement the
 * physics and the limiter.
 *
 * The `Central2DVec` solver class takes two template arguments:
 * `Physics` and `Limiter`.  For `Physics`, we expect the name of a class
 * that defines:
 *
 *  - A type for numerical data (`real`)
 *  - A type for solution and flux vectors in each cell (`vec`)
 *  - A flux computation function (`flux(vec& F, vec& G, const vec& U)`)
 *  - A wave speed computation function
 *    (`wave_speed(real& cx, real& cy, const vec& U)`).
 *
 * The `Limiter` argument is a type with a static function `limdiff`
 * with the signature
 *
 *         limdiff(fm, f0, fp)
 *
 * The semantics are that `fm`, `f0`, and `fp` are three successive
 * grid points in some direction, and the function returns an approximate
 * (scaled) derivative value from these points.
 *
 * The solver keeps arrays for the solution, flux values, derivatives
 * of the solution and the fluxes, and the solution at the next time
 * point.  We use the C++ `vector` class to manage storage for these
 * arrays; but since we want to think of them as 2D arrays, we also
 * provide convenience functions to access them with multiple indices
 * (though we maintain C-style 0-based indexing).  The internal arrays
 * are padded with ghost cells; the ghost cell in the lower left corner
 * of the domain has index (0,0).
 */

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
        u_ (num_fields * nx_all * ny_all),
        f_ (num_fields * nx_all * ny_all),
        g_ (num_fields * nx_all * ny_all),
        ux_(num_fields * nx_all * ny_all),
        uy_(num_fields * nx_all * ny_all),
        fx_(num_fields * nx_all * ny_all),
        gy_(num_fields * nx_all * ny_all),
        v_ (num_fields * nx_all * ny_all) {}

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
    std::vector<real> u_;  // Solution values
    std::vector<real> f_;  // Fluxes in x
    std::vector<real> g_;  // Fluxes in y
    std::vector<real> ux_; // x differences of u
    std::vector<real> uy_; // y differences of u
    std::vector<real> fx_; // x differences of f
    std::vector<real> gy_; // y differences of g
    std::vector<real> v_;  // Solution values at next step

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


/**
 * ## Initialization
 *
 * Before starting the simulation, we need to be able to set the
 * initial conditions.  The `init` function does exactly this by
 * running a callback function at the center of each cell in order
 * to initialize the cell $U$ value.  For the purposes of this function,
 * cell $(i,j)$ is the subdomain
 * $[i \Delta x, (i+1) \Delta x] \times [j \Delta y, (j+1) \Delta y]$.
 */

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

/**
 * ## Time stepper implementation
 *
 * ### Boundary conditions
 *
 * In finite volume methods, boundary conditions are typically applied by
 * setting appropriate values in ghost cells.  For our framework, we will
 * apply periodic boundary conditions; that is, waves that exit one side
 * of the domain will enter from the other side.
 *
 * We apply the conditions by assuming that the cells with coordinates
 * `nghost <= ix <= nx+nghost` and `nghost <= iy <= ny+nghost` are
 * "canonical", and setting the values for all other cells `(ix,iy)`
 * to the corresponding canonical values `(ix+p*nx,iy+q*ny)` for some
 * integers `p` and `q`.
 */

template <class Physics, class Limiter>
void Central2DVec<Physics, Limiter>::apply_periodic() {
    // Copy data between right and left boundaries
    for (int iy = 0; iy < ny_all; ++iy) {
        for (int ix = 0; ix < nghost; ++ix) {
            for (int k = 0; k < num_fields; ++k) {
                u(k, ix,          iy) = uwrap(k, ix,          iy);
                u(k, nx+nghost+ix,iy) = uwrap(k, nx+nghost+ix,iy);
            }
        }
    }

    // Copy data between top and bottom boundaries
    for (int ix = 0; ix < nx_all; ++ix) {
        for (int iy = 0; iy < nghost; ++iy) {
            for (int k = 0; k < num_fields; ++k) {
                u(k, ix,          iy) = uwrap(k, ix,          iy);
                u(k, ix,ny+nghost+iy) = uwrap(k, ix,ny+nghost+iy);
            }
        }
    }
}


/**
 * ### Initial flux and speed computations
 *
 * At the start of each time step, we need the flux values at
 * cell centers (to advance the numerical method) and a bound
 * on the wave speeds in the $x$ and $y$ directions (so that
 * we can choose a time step that respects the specified upper
 * bound on the CFL number).
 */

template <class Physics, class Limiter>
void Central2DVec<Physics, Limiter>::compute_fg_speeds(real& cx_, real& cy_)
{
    using namespace std;
    real cx = 1.0e-15;
    real cy = 1.0e-15;
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            real cell_cx, cell_cy;
            Physics::flux(f(0, ix,iy), f(1, ix, iy), f(2, ix, iy),
                          g(0, ix,iy), g(1, ix, iy), g(2, ix, iy),
                          u(0, ix,iy), u(1, ix, iy), u(2, ix, iy));
            Physics::wave_speed(cell_cx, cell_cy,
                                u(0, ix,iy), u(1, ix, iy), u(2, ix, iy));
            cx = max(cx, cell_cx);
            cy = max(cy, cell_cy);
        }
    cx_ = cx;
    cy_ = cy;
}

/**
 * ### Derivatives with limiters
 *
 * In order to advance the time step, we also need to estimate
 * derivatives of the fluxes and the solution values at each cell.
 * In order to maintain stability, we apply a limiter here.
 */

template <class Physics, class Limiter>
void Central2DVec<Physics, Limiter>::limited_derivs() {
    using L = Limiter;
    for (int iy = 1; iy < ny_all-1; ++iy) {
        for (int ix = 1; ix < nx_all-1; ++ix) {
            for (int k = 0; k < num_fields; ++k) {
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


/**
 * ### Advancing a time step
 *
 * Take one step of the numerical scheme.  This consists of two pieces:
 * a first-order corrector computed at a half time step, which is used
 * to obtain new $F$ and $G$ values; and a corrector step that computes
 * the solution at the full step.  For full details, we refer to the
 * [Jiang and Tadmor paper][jt].
 *
 * The `compute_step` function takes two arguments: the `io` flag
 * which is the time step modulo 2 (0 if even, 1 if odd); and the `dt`
 * flag, which actually determines the time step length.  We need
 * to know the even-vs-odd distinction because the Jiang-Tadmor
 * scheme alternates between a primary grid (on even steps) and a
 * staggered grid (on odd steps).  This means that the data at $(i,j)$
 * in an even step and the data at $(i,j)$ in an odd step represent
 * values at different locations in space, offset by half a space step
 * in each direction.  Every other step, we shift things back by one
 * mesh cell in each direction, essentially resetting to the primary
 * indexing scheme.
 */

template <class Physics, class Limiter>
void Central2DVec<Physics, Limiter>::compute_step(int io, real dt)
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
    for (int j = nghost; j < ny+nghost; ++j){
        for (int i = nghost; i < nx+nghost; ++i){
            for (int k = 0; k < num_fields; ++k) {
                u(k, i, j) = v(k, i-io, j-io);
            }
        }
    }
}


/**
 * ### Advance time
 *
 * The `run` method advances from time 0 (initial conditions) to time
 * `tfinal`.  Note that `run` can be called repeatedly; for example,
 * we might want to advance for a period of time, write out a picture,
 * advance more, and write another picture.  In this sense, `tfinal`
 * should be interpreted as an offset from the time represented by
 * the simulator at the start of the call, rather than as an absolute time.
 *
 * We always take an even number of steps so that the solution
 * at the end lives on the main grid instead of the staggered grid.
 */

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

/**
 * ### Diagnostics
 *
 * The numerical method is supposed to preserve (up to rounding
 * errors) the total volume of water in the domain and the total
 * momentum.  Ideally, we should also not see negative water heights,
 * since that will cause the system of equations to blow up.  For
 * debugging convenience, we'll plan to periodically print diagnostic
 * information about these conserved quantities (and about the range
 * of water heights).
 */

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

//ldoc off
#endif /* CENTRAL2D_VEC_H*/
