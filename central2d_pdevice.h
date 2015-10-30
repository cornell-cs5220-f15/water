#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#pragma offload_attribute(push,target(mic))
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <memory>
#include <omp.h>
#pragma offload_attribute(pop)

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
 * The `Central2D` solver class takes two template arguments:
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
class Central2D {
public:
    typedef typename Physics::real real;
    typedef typename Physics::vec  vec;

    Central2D(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              int nxblocks = 1,   // Number of blocks in x for batching
              int nyblocks = 1,   // Number of blocks in y for batching
              int nbatch = 1,     // Number of timesteps to batch per block
              real cfl = 0.45) :  // Max allowed CFL number
        nx(nx), ny(ny), nxblocks(nxblocks), nyblocks(nyblocks),
        nbatch(nbatch), nghost(1+nbatch*2), // Number of ghost cells depend on batch size
        nx_all(nx + 2*nghost),
        ny_all(ny + 2*nghost),
        nthreads(nxblocks*nyblocks),
        dx(w/nx), dy(h/ny),
        cfl(cfl),
        u_(nx_all * ny_all) {}

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

    // Read / write elements of simulation state
    vec&       operator()(int i, int j) {
        return u_[(j+nghost)*nx_all+(i+nghost)];
    }

    const vec& operator()(int i, int j) const {
        return u_[(j+nghost)*nx_all+(i+nghost)];
    }

private:

    // Simulator parameters to offload to device
    typedef struct {
      int nghost;
      int nx, ny;
      int nxblocks, nyblocks;
      int nbatch;
      int nthreads;
      int nx_all, ny_all;
      real dx, dy;
      real cfl;
    } Parameters;

    // Class for encapsulating per-thread local state
    class LocalState {
     public:
      __declspec(target(mic))
      LocalState(int nx, int ny) :
        nx(nx), ny(ny),
        u_ (nx * ny),
        v_ (nx * ny),
        f_ (nx * ny),
        g_ (nx * ny),
        ux_(nx * ny),
        uy_(nx * ny),
        fx_(nx * ny),
        gy_(nx * ny) {}

      __declspec(target(mic))
      ~LocalState() {}

      // Array accessor functions
      __declspec(target(mic)) vec& u(int ix, int iy)  { return u_[offset(ix,iy)]; }
      __declspec(target(mic)) vec& v(int ix, int iy)  { return v_[offset(ix,iy)]; }
      __declspec(target(mic)) vec& f(int ix, int iy)  { return f_[offset(ix,iy)]; }
      __declspec(target(mic)) vec& g(int ix, int iy)  { return g_[offset(ix,iy)]; }
      __declspec(target(mic)) vec& ux(int ix, int iy) { return ux_[offset(ix,iy)]; }
      __declspec(target(mic)) vec& uy(int ix, int iy) { return uy_[offset(ix,iy)]; }
      __declspec(target(mic)) vec& fx(int ix, int iy) { return fx_[offset(ix,iy)]; }
      __declspec(target(mic)) vec& gy(int ix, int iy) { return gy_[offset(ix,iy)]; }

      // Miscellaneous accessors
      __declspec(target(mic)) int get_nx() { return nx; }
      __declspec(target(mic)) int get_ny() { return ny; }

     private:
      // Helper to calculate 1D offset from 2D coordinates
      __declspec(target(mic))
      int offset(int ix, int iy) const { return iy*nx+ix; }

      const int nx, ny;

      std::vector<vec> u_;  // Solution values
      std::vector<vec> v_;  // Solution values at next step
      std::vector<vec> f_;  // Fluxes in x
      std::vector<vec> g_;  // Fluxes in y
      std::vector<vec> ux_; // x differences of u
      std::vector<vec> uy_; // y differences of u
      std::vector<vec> fx_; // x differences of f
      std::vector<vec> gy_; // y differences of g
    };

    const int nghost;             // Number of ghost cells
    const int nx, ny;             // Number of (non-ghost) cells in x/y
    const int nxblocks, nyblocks; // Number of blocks for batching in x/y
    const int nbatch;             // Number of timesteps to batch per block
    const int nthreads;           // Number of threads
    const int nx_all, ny_all;     // Total cells in x/y (including ghost)
    const real dx, dy;            // Cell size in x/y
    const real cfl;               // Allowed CFL number

    // Global solution values
    std::vector<vec> u_;

    // Array accessor function

    __declspec(target(mic))
    int offset(Parameters params, int ix, int iy) const {
        return iy*params.nx_all+ix;
    }

    vec& u(int ix, int iy) { return u_[iy*nx_all+ix]; }

    // Wrapped accessor (periodic BC)
    __declspec(target(mic))
    int ioffset(Parameters params, int ix, int iy) {
        return offset( params,
                       (ix+params.nx-params.nghost) % params.nx + params.nghost,
                       (iy+params.ny-params.nghost) % params.ny + params.nghost );
    }

//    vec& uwrap(int ix, int iy) { return u_[ioffset(ix,iy)]; }

    // Initialize per-thread local state inside of offloaded kernel
    __declspec(target(mic)) void init_locals(
        Parameters params, std::vector<LocalState*>& locals);

    // Apply limiter to all components in a vector
    __declspec(target(mic))
    static void limdiff(vec& du, const vec& um, const vec& u0, const vec& up) {
        for (int m = 0; m < du.size(); ++m)
            du[m] = Limiter::limdiff(um[m], u0[m], up[m]);
    }

    // Stages of the main algorithm
    __declspec(target(mic)) void apply_periodic(Parameters params, real* u);
    __declspec(target(mic)) void compute_wave_speeds(Parameters params, real& cx, real& cy, real* u);
    __declspec(target(mic)) void compute_flux(Parameters params, LocalState* local);
    __declspec(target(mic)) void limited_derivs(Parameters params, LocalState* local);
    __declspec(target(mic)) void compute_step(Parameters params, LocalState* local, int io, real dt);

    // Copy data to and from local buffers
    __declspec(target(mic)) void copy_to_local(Parameters params, LocalState* local, int tid, real* u);
    __declspec(target(mic)) void copy_from_local(Parameters params, LocalState* local, int tid, real* u);

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
void Central2D<Physics, Limiter>::init(F f)
{
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix)
            f(u(nghost+ix,nghost+iy), (ix+0.5)*dx, (iy+0.5)*dy);
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::init_locals(
    Parameters params, std::vector<LocalState*>& locals)
{
    // Dimensions of block assigned to each thread
    int nx_per_block = ceil(params.nx / (real)params.nxblocks);
    int ny_per_block = ceil(params.ny / (real)params.nyblocks);

    // Number of elements beyond grid boundary if block dimensions do
    // not evenly divide the grid dimensions.
    int nx_overhang = (nx_per_block * params.nxblocks) - params.nx;
    int ny_overhang = (ny_per_block * params.nyblocks) - params.ny;

    assert( nx_overhang >= 0 && ny_overhang >= 0 );

    // Dimensions of block with ghost cells
    int nx_per_block_padded = nx_per_block + 2*params.nghost;
    int ny_per_block_padded = ny_per_block + 2*params.nghost;

    // Set dimensions of each block. Block dimensions are only
    // different if they do not evenly divide the grid dimensions at
    // the boundaries. In such cases, we need to subtract the
    // overhang count from the corresponding dimension for the
    // boundary blocks.
    for (int j = 0; j < params.nyblocks; ++j) {
        int ny_local = (j == params.nyblocks - 1) ? ny_per_block_padded - ny_overhang
                     :                              ny_per_block_padded;
        for (int i = 0; i < params.nxblocks; ++i) {
            int nx_local = (i == params.nxblocks - 1) ? nx_per_block_padded - nx_overhang
                         :                              nx_per_block_padded;
            locals.push_back(new LocalState(nx_local, ny_local));
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
void Central2D<Physics, Limiter>::apply_periodic(Parameters params, real* u)
{
    // Copy data between right and left boundaries
    for (int iy = 0; iy < params.ny_all; ++iy)
        for (int ix = 0; ix < params.nghost; ++ix) {
            int iu      = offset(params, ix, iy) * 3;
            int iu_wrap = ioffset(params, ix, iy) * 3;
            u[iu+0] = u[iu_wrap+0];
            u[iu+1] = u[iu_wrap+1];
            u[iu+2] = u[iu_wrap+2];

            iu      = offset(params, params.nx+params.nghost+ix, iy) * 3;
            iu_wrap = ioffset(params, params.nx+params.nghost+ix, iy) * 3;
            u[iu+0] = u[iu_wrap+0];
            u[iu+1] = u[iu_wrap+1];
            u[iu+2] = u[iu_wrap+2];
        }

    // Copy data between top and bottom boundaries
    for (int ix = 0; ix < params.nx_all; ++ix)
        for (int iy = 0; iy < params.nghost; ++iy) {
            int iu      = offset(params, ix, iy) * 3;
            int iu_wrap = ioffset(params, ix, iy) * 3;
            u[iu+0] = u[iu_wrap+0];
            u[iu+1] = u[iu_wrap+1];
            u[iu+2] = u[iu_wrap+2];

            iu      = offset(params, ix, params.ny+params.nghost+iy) * 3;
            iu_wrap = ioffset(params, ix, params.ny+params.nghost+iy) * 3;
            u[iu+0] = u[iu_wrap+0];
            u[iu+1] = u[iu_wrap+1];
            u[iu+2] = u[iu_wrap+2];
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
void Central2D<Physics, Limiter>::compute_wave_speeds(
    Parameters params, real& cx_, real& cy_, real* u)
{
    real cx = 1.0e-15;
    real cy = 1.0e-15;
    for (int iy = 0; iy < params.ny_all; ++iy)
        for (int ix = 0; ix < params.nx_all; ++ix) {
            real cell_cx, cell_cy;
            int iu = offset(params, ix, iy) * 3;
            vec u_tmp = {u[iu+0], u[iu+1], u[iu+2]};
            Physics::wave_speed(cell_cx, cell_cy, u_tmp);
            cx = std::max(cx, cell_cx);
            cy = std::max(cy, cell_cy);
        }
    cx_ = cx;
    cy_ = cy;
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::compute_flux(
    Parameters params, LocalState* local)
{
    int ny_per_block = local->get_ny();
    int nx_per_block = local->get_nx();

    for (int iy = 0; iy < ny_per_block; ++iy)
      #pragma ivdep
        for (int ix = 0; ix < nx_per_block; ++ix)
            Physics::flux(local->f(ix,iy),
                          local->g(ix,iy),
                          local->u(ix,iy));
}

/**
 * ### Derivatives with limiters
 *
 * In order to advance the time step, we also need to estimate
 * derivatives of the fluxes and the solution values at each cell.
 * In order to maintain stability, we apply a limiter here.
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::limited_derivs(
    Parameters params, LocalState* local)
{
    int ny_per_block = local->get_ny();
    int nx_per_block = local->get_nx();

    for (int iy = 1; iy < ny_per_block-1; ++iy)
        #pragma ivdep
        for (int ix = 1; ix < nx_per_block-1; ++ix) {

            // x derivs
            limdiff( local->ux(ix,iy), local->u(ix-1,iy),
                     local->u(ix,iy),  local->u(ix+1,iy) );
            limdiff( local->fx(ix,iy), local->f(ix-1,iy),
                     local->f(ix,iy),  local->f(ix+1,iy) );

            // y derivs
            limdiff( local->uy(ix,iy), local->u(ix,iy-1),
                     local->u(ix,iy),  local->u(ix,iy+1) );
            limdiff( local->gy(ix,iy), local->g(ix,iy-1),
                     local->g(ix,iy),  local->g(ix,iy+1) );
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
void Central2D<Physics, Limiter>::compute_step(
    Parameters params, LocalState* local, int io, real dt)
{
    int ny_per_block = local->get_ny();
    int nx_per_block = local->get_nx();

    real dtcdx2 = 0.5 * dt / params.dx;
    real dtcdy2 = 0.5 * dt / params.dy;

    // Predictor (flux values of f and g at half step)
    for (int iy = 1; iy < ny_per_block-1; ++iy)
        for (int ix = 1; ix < nx_per_block-1; ++ix) {
            vec uh = local->u(ix,iy);
            for (int m = 0; m < uh.size(); ++m) {
                uh[m] -= dtcdx2 * local->fx(ix,iy)[m];
                uh[m] -= dtcdy2 * local->gy(ix,iy)[m];
            }
            Physics::flux(local->f(ix,iy), local->g(ix,iy), uh);
        }

    // Corrector (finish the step)
    for (int iy = params.nghost-io; iy < ny_per_block-params.nghost-io; ++iy)
        for (int ix = params.nghost-io; ix < nx_per_block-params.nghost-io; ++ix) {
            for (int m = 0; m < local->v(ix,iy).size(); ++m) {

                real u_sum = local->u(ix,iy)[m]
                           + local->u(ix+1,iy)[m]
                           + local->u(ix,iy+1)[m]
                           + local->u(ix+1,iy+1)[m];

                real uxy_sum = local->ux(ix+1,iy)[m]
                             - local->ux(ix,iy)[m]
                             + local->ux(ix+1,iy+1)[m]
                             - local->ux(ix,iy+1)[m]
                             + local->uy(ix,iy+1)[m]
                             - local->uy(ix,iy)[m]
                             + local->uy(ix+1,iy+1)[m]
                             - local->uy(ix+1,iy)[m];

                real f_sum = local->f(ix+1,iy)[m]
                           - local->f(ix,iy)[m]
                           + local->f(ix+1,iy+1)[m]
                           - local->f(ix,iy+1)[m];

                real g_sum = local->g(ix,iy+1)[m]
                           - local->g(ix,iy)[m]
                           + local->g(ix+1,iy+1)[m]
                           - local->g(ix+1,iy)[m];

                local->v(ix,iy)[m] = 0.2500 * u_sum
                                          - 0.0625 * uxy_sum
                                          - dtcdx2 * f_sum
                                          - dtcdy2 * g_sum;
            }
        }

    // Copy from v storage back to main grid
    for (int j = params.nghost; j < ny_per_block-params.nghost; ++j)
        for (int i = params.nghost; i < nx_per_block-params.nghost; ++i)
            local->u(i,j) = local->v(i-io,j-io);

}

/**
 * ### Copy to/from local buffers
 *
 * Each thread needs its own local view of the solution and flux vectors
 * in order to enable batching of multiple time steps. After every
 * synchronization point, the relevant blocks of the global solution
 * vectors are copied to the per-thread local buffers. Before the next
 * synchronization point, each thread copies its locally updated
 * solutions to the global solution vectors.
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::copy_to_local(
    Parameters params, LocalState* local, int tid, real* u)
{
    int ny_per_block = local->get_ny();
    int nx_per_block = local->get_nx();

    int biy     = tid / params.nxblocks;
    int bix     = tid % params.nxblocks;
    int biy_off = biy * (ny_per_block - 2*params.nghost);
    int bix_off = bix * (nx_per_block - 2*params.nghost);

    for (int iy = 0; iy < ny_per_block; ++iy)
        for (int ix = 0; ix < nx_per_block; ++ix) {
            int iu = ((biy_off+iy)*params.nx_all + bix_off+ix) * 3;
            local->u(ix, iy)[0] = u[iu+0]; // h
            local->u(ix, iy)[1] = u[iu+1]; // hu
            local->u(ix, iy)[2] = u[iu+2]; // hv
        }
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::copy_from_local(
    Parameters params, LocalState* local, int tid, real* u)
{
    int ny_per_block = local->get_ny();
    int nx_per_block = local->get_nx();

    int biy     = tid / params.nxblocks;
    int bix     = tid % params.nxblocks;
    int biy_off = biy * (ny_per_block - 2*params.nghost);
    int bix_off = bix * (nx_per_block - 2*params.nghost);

    for (int iy = params.nghost; iy < ny_per_block - params.nghost; ++iy)
        for (int ix = params.nghost; ix < nx_per_block - params.nghost; ++ix) {
            int iu = ((biy_off+iy)*params.nx_all+bix_off+ix) * 3;
            u[iu+0] = local->u(ix, iy)[0]; // h
            u[iu+1] = local->u(ix, iy)[1]; // hu
            u[iu+2] = local->u(ix, iy)[2]; // hv
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
void Central2D<Physics, Limiter>::run(real tfinal)
{
    // Offload computation to MIC

    real* u_offload      = reinterpret_cast<real*>(u_.data());
    int   u_offload_size = u_.size() * 3;

    #pragma offload target(mic:0) in(nghost) in(nx) in(ny) in(nxblocks) in(nyblocks) \
                                  in(nbatch) in(nthreads) in(nx_all) in(ny_all) \
                                  in(dx) in(dy) in(cfl) in(tfinal) \
                                  inout(u_offload : length(u_offload_size))
    {

    // Copy parameters from host to device
    Parameters params;
    params.nghost   = nghost;
    params.nx       = nx;
    params.ny       = ny;
    params.nxblocks = nxblocks;
    params.nyblocks = nyblocks;
    params.nbatch   = nbatch;
    params.nthreads = nthreads;
    params.nx_all   = nx_all;
    params.ny_all   = ny_all;
    params.dx       = dx;
    params.dy       = dy;
    params.cfl      = cfl;

    // Initialize per-thread local buffers on the device
    std::vector<LocalState*> locals;
    init_locals(params, locals);

    // Main computation loop
    bool done = false;
    real t = 0;
    while (!done) {

        // We only need to update the ghost cells after all threads have
        // exhausted valid data in the ghost cells in order to calculate
        // the number of steps in the batch.
        apply_periodic(params, u_offload);

        // We only need to calculate the wave speeds at the beginning of
        // each super-step to determine the dt for both the even/odd
        // sub-steps.
        real cx, cy;
        compute_wave_speeds(params, cx, cy, u_offload);

        // Break out of the loop after this super-step if we have
        // simulated at least tfinal seconds.
        real dt = params.cfl / std::max(cx/params.dx, cy/params.dy);
        int  modified_nbatch = params.nbatch;
        if (t + 2*params.nbatch*dt >= tfinal) {
            modified_nbatch = ceil((tfinal-t)/(2*dt));
            done = true;
        }

        // Parallelize computation across partitioned blocks
        #pragma omp parallel num_threads(params.nthreads)
        {
          int tid = omp_get_thread_num();

          // Copy global data to local buffers
          copy_to_local(params, locals[tid], tid, u_offload);

          // Batch multiple timesteps
          for (int bi = 0; bi < modified_nbatch; ++bi) {

              // Execute the even and odd sub-steps for each super-step
              for (int io = 0; io < 2; ++io) {
                  compute_flux(params, locals[tid]);
                  limited_derivs(params, locals[tid]);
                  compute_step(params, locals[tid], io, dt);
              }
          }

          // Copy local data to global buffer
          copy_from_local(params, locals[tid], tid, u_offload);
        }

        // Update simulated time
        t += 2*modified_nbatch*dt;
    }

    // Clean up local state
    for (auto local : locals)
      free( local );
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
void Central2D<Physics, Limiter>::solution_check()
{
    using namespace std;
    real h_sum = 0, hu_sum = 0, hv_sum = 0;
    real hmin = u(nghost,nghost)[0];
    real hmax = hmin;
    for (int j = nghost; j < ny+nghost; ++j)
        for (int i = nghost; i < nx+nghost; ++i) {
            vec& uij = u(i,j);
            real h = uij[0];
            h_sum += h;
            hu_sum += uij[1];
            hv_sum += uij[2];
            hmax = max(h, hmax);
            hmin = min(h, hmin);
            assert( h > 0) ;
        }
    real cell_area = dx*dy;
    h_sum *= cell_area;
    hu_sum *= cell_area;
    hv_sum *= cell_area;
    printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n",
           h_sum, hu_sum, hv_sum, hmin, hmax);
}

//ldoc off
#endif /* CENTRAL2D_H*/
