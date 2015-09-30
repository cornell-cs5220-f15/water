//ldoc on
/**
 * % Shallow water simulation
 * % David Bindel
 * % 2015-09-30
 * 
 * This code implements the [Jiang-Tadmor centered finite volume
 * scheme][jt] for conservation law PDEs in 2D, and uses the shallow
 * water wave equations as an example.  As implemented, the code is
 * serial, and not particularly carefully tuned.  Your goal: make the
 * code run fast on the cluster, using OpenMP and (ideally) taking
 * advantage of the Xeon Phi accelerators.
 * 
 * While I have tried not to do anything too obscure, this code does
 * use some C++ 11 features.  If you want to build on your own
 * machine, you may need to figure out the flag needed to tell your
 * compiler that you are using this C++ dialect.
 * 
 */
//ldoc off

#include <cstdio>
#include <cmath>
#include <cassert>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <array>

using namespace std;

//ldoc on
/**
 * # Shallow water equations
 * 
 * ## Physics picture
 * 
 * The shallow water equations are a two-dimensional PDE system
 * that describes water waves that are very long compared to the
 * water depth.  It applies even in situations that you might not
 * think of as "shallow"; for example, tsunami waves are long enough
 * that they can be modeled using the shallow water equations even
 * when traveling over mile-deep parts of oceans.
 * 
 * The shallow water equations treat water as incompressible and
 * inviscid, and assume that the horizontal velocity remains constant
 * in any vertical column of water.  The unknowns at each point are
 * the water height and the total horizontal momentum in a water
 * column; the equations describe conservation of mass (fluid is
 * neither created nor destroyed) and conservation of linear momentum.
 * We will solve these equations with a numerical method that also
 * exactly conserves mass and momentum (up to rounding error), though
 * it only approximately conserves energy.
 * 
 * I was inspired to use this system for our assignment by reading the
 * chapter on [shallow water simulation in MATLAB][exm] from Cleve
 * Moler's books on "Experiments in MATLAB"; there is also a very readable
 * [Wikipedia article][wiki] on the shallow water equations.
 * The basic variables are water height ($h$), and the velocity components
 * ($u, v$).  We write the governing equations in the form
 * $$
 *   U_t = F(U)_x + G(U)_y
 * $$
 * where
 * $$
 *   U = \begin{bmatrix} h \\ hu \\ hv \end{bmatrix},
 *   F = \begin{bmatrix} hu \\ h^2 u + gh^2/2 \\ huv \end{bmatrix}
 *   G = \begin{bmatrix} hv \\ huv \\ h^2 v + gh^2/2 \end{bmatrix}
 * $$
 * The functions $F$ and $G$ are called *fluxes*, and describe how the
 * conserved quantities (volume and momentum) enter and exit a region
 * of space.
 * 
 * Note that we also need a bound on the characteristic wave speeds
 * for the problem in order to ensure that our method doesn't explode;
 * we use this to control the Courant-Friedrichs-Levy (CFL) number
 * relating wave speeds, time steps, and space steps.  For the shallow
 * water equations, the characteristic wave speed is $\sqrt{g h}$
 * where $g$ is the gravitational constant and $h$ is the height of the
 * water; in addition, we have to take into account the velocity of
 * the underlying flow.
 * 
 * [exm]: https://www.mathworks.com/moler/exm/chapters/water.pdf
 * [wiki]: https://en.wikipedia.org/wiki/Shallow_water_equations
 * 
 * ## Implementation
 * 
 * Our solver takes advantage of C++ templates to get (potentially)
 * good performance while keeping a clean abstraction between the
 * solver code and the details of the physics.  The `Shallow2D`
 * class specifies the precision of the comptutation (single precision),
 * the data type used to represent vectors of unknowns and fluxes
 * (the C++ `std::array`).  We are really only using the class as 
 * name space; we never create an instance of type `Shallow2D`,
 * and the `flux` and `wave_speed` functions needed by the solver are
 * declared as static (and inline, in the hopes of getting the compiler
 * to optimize for us).
 */

struct Shallow2D {

    // Type parameters for solver
    typedef float real;
    typedef std::array<real,3> vec;

    // Gravitational force (compile time constant)
    static constexpr real g = 9.8;

    // Compute shallow water fluxes F(U), G(U)
    static void flux(vec& FU, vec& GU, const vec& U) {
        real h = U[0], hu = U[1], hv = U[2];

        FU[0] = hu;
        FU[1] = hu*hu/h + (0.5*g)*h*h;
        FU[2] = hu*hv/h;

        GU[0] = hv;
        GU[1] = hu*hv/h;
        GU[2] = hv*hv/h + (0.5*g)*h*h;
    }

    // Compute shallow water wave speed
    static void wave_speed(real& cx, real& cy, const vec& U) {
        real h = U[0], hu = U[1], hv = U[2];
        real root_gh = sqrt(g * h);  // NB: Don't let h go negative!
        cx = abs(hu/h) + root_gh;
        cy = abs(hv/h) + root_gh;
    }
};


/**
 * # MinMod limiter
 * 
 * Numerical methods for solving nonlinear wave equations are
 * complicated by the fact that even with smooth initial data, a
 * nonlinear wave can develop discontinuities (shocks) in finite time.
 * 
 * This makes for interesting analysis, since a "strong" solution
 * that satisfies the differential equation no longer makes sense at
 * a shock -- instead, we have to come up with some mathematically
 * and physically reasonable definition of a "weak" solution that
 * satisfies the PDE away from the shock and satisfies some other
 * condition (an entropy condition) at the shock.
 * 
 * The presence of shocks also makes for interesting *numerical*
 * analysis, because we need to be careful about employing numerical
 * differentiation formulas that sample a discontinuous function at
 * points on different sides of a shock.  Using such formulas naively
 * usually causes the numerical method to become unstable.  A better
 * method -- better even in the absence of shocks! -- is to consider
 * multiple numerical differentiation formulas and use the highest
 * order one that "looks reasonable" in the sense that it doesn't
 * predict wildly larger slopes than the others.  Because these
 * combined formulas *limit* the wild behavior of derivative estimates
 * across a shock, we call them *limiters*.  With an appropriate limiter,
 * we can construct methods that have high-order accuracy away from shocks
 * and are at least first-order accurate close to a shock.  These are
 * sometimes called *high-resolution* methods.
 *
 * The MinMod (minimum modulus) limiter is one example of a limiter.
 * The MinMod limiter estimates the slope through points $f_-, f_0, f_+$
 * (with the step $h$ scaled to 1) by
 * $$
 *   f' = \operatorname{minmod}((f_+-f_-)/2, \theta(f_+-f_0), \theta(f_0-f_-))
 * $$
 * where the minmod function returns the argument with smallest absolute
 * value if all arguments have the same sign, and zero otherwise.
 * Common choices of $\theta$ are $\theta = 1.0$ and $\theta = 2.0$.
 * 
 * The minmod limiter *looks* like it should be expensive to computer,
 * since superficially it seems to require a number of branches.
 * We do something a little tricky, getting rid of the condition
 * on the sign of the arguments using the `copysign` instruction.
 * If the compiler does the "right" thing with `max` and `min`
 * for floating point arguments (translating them to branch-free
 * intrinsic operations), this implementation should be relatively fast.
 * 
 * There are many other potential choices of limiters as well.  We'll
 * stick with this one for the code, but you should feel free to
 * experiment with others if you know what you're doing and think it
 * will improve performance or accuracy.
 */

template <class real>
struct MinMod {
    static constexpr real theta = 2.0;

    // Branch-free computation of minmod of two numbers
    static real xmin(real a, real b) {
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
 * The Jiang-Tadmor scheme works by alternating between a main grid
 * and a staggered grid offset by half a step in each direction.
 * We currently manage this implicitly: the arrays at even time steps
 * represent cell values on the main grid, and arrays at odd steps
 * represent cell values on the staggered grid.  Our main `run` 
 * function always takes an even number of time steps to ensure we end
 * up on the primary grid.
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

template <class Physics, typename Limiter>
class Central2D {
public:
    typedef typename Physics::real real;
    typedef typename Physics::vec  vec;

    Central2D(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              real cfl = 0.8) :   // Max allowed CFL number
        nx(nx), ny(ny),
        nx_all(nx + 2*nghost),
        ny_all(ny + 2*nghost),
        dx(w/nx), dy(h/ny),
        cfl(cfl), 
        u_ (nx_all * ny_all),
        f_ (nx_all * ny_all),
        g_ (nx_all * ny_all),
        ux_(nx_all * ny_all),
        uy_(nx_all * ny_all),
        fx_(nx_all * ny_all),
        gy_(nx_all * ny_all),
        v_ (nx_all * ny_all) {}

    // Advance from time 0 to time tfinal
    void run(real tfinal);

    // Advance in nsteps steps of time tframe, writing a data frame
    // for the visualizer after each.
    void run_viz(real tframe, int nsteps);

    // Call f(Uxy, x, y) at each cell center to set initial conditions
    template <typename F>
    void init(F f);

    // Write a gray-scale image file with f(U(x,y)) in [0,255]
    //  as the gray level for the cell centered at (x,y)
    template <typename F>
    void write_pgm(const char* name, F f);

    // Write current state to a file FP for use with the
    //   Python visualizer.
    void write_viz(FILE* fp);

    // Diagnostics
    void solution_check();

private:
    static constexpr int nghost = 3;   // Number of ghost cells

    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number

    vector<vec> u_;            // Solution values
    vector<vec> f_;            // Fluxes in x
    vector<vec> g_;            // Fluxes in y
    vector<vec> ux_;           // x differences of u
    vector<vec> uy_;           // y differences of u
    vector<vec> fx_;           // x differences of f
    vector<vec> gy_;           // y differences of g
    vector<vec> v_;            // Solution values at next step

    // Array accessor functions

    int offset(int ix, int iy)  { return iy*nx_all+ix; }

    vec& u(int ix, int iy)    { return u_[offset(ix,iy)]; }
    vec& v(int ix, int iy)    { return v_[offset(ix,iy)]; }
    vec& f(int ix, int iy)    { return f_[offset(ix,iy)]; }
    vec& g(int ix, int iy)    { return g_[offset(ix,iy)]; }

    vec& ux(int ix, int iy)   { return ux_[offset(ix,iy)]; }
    vec& uy(int ix, int iy)   { return uy_[offset(ix,iy)]; }
    vec& fx(int ix, int iy)   { return fx_[offset(ix,iy)]; }
    vec& gy(int ix, int iy)   { return gy_[offset(ix,iy)]; }

    // Wrapped accessor (periodic BC)
    int ioffset(int ix, int iy) {
        return offset( (ix+nx-nghost) % nx + nghost,
                       (iy+ny-nghost) % ny + nghost );
    }

    vec& uwrap(int ix, int iy)  { return u_[ioffset(ix,iy)]; }

    // Apply limiter to all components in a vector
    static void limdiff(vec& du, const vec& um, const vec& u0, const vec& up) {
        for (int m = 0; m < du.size(); ++m)
            du[m] = Limiter::limdiff(um[m], u0[m], up[m]);
    }

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
void Central2D<Physics, Limiter>::init(F f)
{
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix)
            f(u(nghost+ix,nghost+iy), (ix+0.5)*dx, (iy+0.5)*dy);
}

/**
 * ## I/O
 * 
 * After finishing a run (or every several steps), we might want to
 * write out a data file for post processing.  One simple approach is
 * to draw a gray scale or color picture showing some scalar quantity
 * at each point.  The Portable Gray Map (PGM) format is one of the
 * few graphics formats that can be dumped out in a handful of lines
 * of code without any library calls.  The files can be converted to
 * something more modern and snazzy (like a PNG or GIF) later on.
 * Note that we don't actually dump out the state vector for each cell
 * -- we need to produce something that is an integer in the range
 * [0,255].  That's what the function `f` is for!
 */

template <class Physics, class Limiter>
template <typename F>
void Central2D<Physics, Limiter>::write_pgm(const char* fname, F f)
{
    FILE* fp = fopen(fname, "wb");
    fprintf(fp, "P5\n");
    fprintf(fp, "%d %d 255\n", nx, ny);
    for (int iy = ny-1; iy >= 0; --iy)
        for (int ix = 0; ix < nx; ++ix)
            fputc(min(255, max(0, f(u(ix+nghost,iy+nghost)))), fp);
    fclose(fp);
}

/**
 * 
 * An alternative to writing an image file is to write a data file for
 * further processing by some other program -- in this case, a Python
 * visualizer.
 * 
 */

// TODO: This should really be a binary file format!
template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::write_viz(FILE* fp)
{
    for (int j = nghost; j < ny+nghost; ++j)
        for (int i = nghost; i < nx+nghost; ++i)
            fprintf(fp, "%f,", u(i,j)[0]);
    fprintf(fp, "\n");
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
void Central2D<Physics, Limiter>::apply_periodic()
{
    // Copy data between right and left boundaries
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nghost; ++ix) {
            u(ix,          iy) = uwrap(ix,          iy);
            u(nx+nghost+ix,iy) = uwrap(nx+nghost+ix,iy);
        }

    // Copy data between top and bottom boundaries
    for (int ix = 0; ix < nx_all; ++ix)
        for (int iy = 0; iy < nghost; ++iy) {
            u(ix,          iy) = uwrap(ix,          iy);
            u(ix,ny+nghost+iy) = uwrap(ix,ny+nghost+iy);
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
void Central2D<Physics, Limiter>::compute_fg_speeds(real& cx_, real& cy_)
{
    real cx = 1.0e-15;
    real cy = 1.0e-15;
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            real cell_cx, cell_cy;
            Physics::flux(f(ix,iy), g(ix,iy), u(ix,iy));
            Physics::wave_speed(cell_cx, cell_cy, u(ix,iy));
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
void Central2D<Physics, Limiter>::limited_derivs()
{
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {

            // x derivs
            limdiff( ux(ix,iy), u(ix-1,iy), u(ix,iy), u(ix+1,iy) );
            limdiff( fx(ix,iy), f(ix-1,iy), f(ix,iy), f(ix+1,iy) );

            // y derivs
            limdiff( uy(ix,iy), u(ix,iy-1), u(ix,iy), u(ix,iy+1) );
            limdiff( gy(ix,iy), g(ix,iy-1), g(ix,iy), g(ix,iy+1) );
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
void Central2D<Physics, Limiter>::compute_step(int io, real dt)
{
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;

    // Predictor (flux values of f and g at half step)
    for (int iy = 1; iy < ny_all-1; ++iy)
        for (int ix = 1; ix < nx_all-1; ++ix) {
            vec uh = u(ix,iy);
            for (int m = 0; m < uh.size(); ++m) {
                uh[m] -= dtcdx2 * fx(ix,iy)[m];
                uh[m] -= dtcdy2 * gy(ix,iy)[m];
            }
            Physics::flux(f(ix,iy), g(ix,iy), uh);
        }

    // Corrector (finish the step)
    for (int iy = nghost-io; iy < ny+nghost-io; ++iy)
        for (int ix = nghost-io; ix < nx+nghost-io; ++ix) {
            for (int m = 0; m < v(ix,iy).size(); ++m) {
                v(ix,iy)[m] =
                    0.2500 * ( u(ix,  iy)[m] + u(ix+1,iy  )[m] +
                               u(ix,iy+1)[m] + u(ix+1,iy+1)[m] ) -
                    0.0625 * ( ux(ix+1,iy  )[m] - ux(ix,iy  )[m] +
                               ux(ix+1,iy+1)[m] - ux(ix,iy+1)[m] +
                               uy(ix,  iy+1)[m] - uy(ix,  iy)[m] +
                               uy(ix+1,iy+1)[m] - uy(ix+1,iy)[m] ) -
                    dtcdx2 * ( f(ix+1,iy  )[m] - f(ix,iy  )[m] +
                               f(ix+1,iy+1)[m] - f(ix,iy+1)[m] ) -
                    dtcdy2 * ( g(ix,  iy+1)[m] - g(ix,  iy)[m] +
                               g(ix+1,iy+1)[m] - g(ix+1,iy)[m] );
            }
        }

    // Copy from v storage back to main grid
    for (int j = nghost; j < ny+nghost; ++j){
        for (int i = nghost; i < nx+nghost; ++i){
            u(i,j) = v(i-io,j-io);
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
void Central2D<Physics, Limiter>::run(real tfinal)
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
                dt = cfl / max(cx/dx, cy/dy);
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
 * The `run_viz` routine writes data frames to an output file at
 * regular time intervals.  Frames are generated every `tframe` time
 * units, and `nsteps+1` frames are generated in total (including one
 * for the initial conditions).
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::run_viz(real tframe, int nsteps)
{
    FILE *fp = fopen("waves.txt", "w");
    solution_check();
    write_viz(fp);
    for (int i = 0; i < nsteps; ++i) {
        run(tframe);
        solution_check();
        write_viz(fp);
    }
    fclose(fp);
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


/**
 * # Driver routines
 * 
 * For the driver, we need to put everything together: we're running
 * a `Central2D` solver for the `Shallow2D` physics with a `MinMod`
 * limiter.
 */

typedef Central2D< Shallow2D, MinMod<Shallow2D::real> > Sim;

/**
 * ## Initial states and graphics
 * 
 * The following functions define some interesting initial conditions.
 * Ideally, I would be doing this via a Python interface.  But I
 * couldn't be bothered to deal with the linker.
 */

// Circular dam break problem
void dam_break(Sim::vec& u, double x, double y)
{
    x -= 1;
    y -= 1;
    u[0] = 1.0 + 0.5*(x*x + y*y < 0.25+1e-5);
    u[1] = 0;
    u[2] = 0;
}

// Still pond (ideally, nothing should move here!)
void pond(Sim::vec& u, double x, double y)
{
    u[0] = 1.0;
    u[1] = 0;
    u[2] = 0;
}

// River (ideally, the solver shouldn't do much with this, either)
void pond(Sim::vec& u, double x, double y)
{
    u[0] = 1.0;
    u[1] = 1.0;
    u[2] = 0;
}


/**
 * ## Summary plots
 * 
 * We can plot either height or (total) momentum as interesting
 * scalar quantities.  The ranges (0 to 3.0 and 0 to 2.5) are
 * completely made up -- it would probably be smarter to change
 * those!
 */

int show_height(const Sim::vec& u)
{
    return 255 * (u[0] / 3.0);
}

int show_momentum(const Sim::vec& u)
{
    return 255 * sqrt(u[1]*u[1] + u[2]*u[2]) / 2.5;
}


/**
 * # Main driver
 * 
 * Again, this should really invoke an option parser, or be glued
 * to an interface in some appropriate scripting language (Python,
 * or perhaps Lua).
 */

int main()
{
    Sim sim(2,2, 200,200, 0.2);
    sim.init(dam_break);
    sim.solution_check();
    sim.write_pgm("test.pgm", show_height);
    sim.run(0.5);
    sim.solution_check();
    sim.write_pgm("test2.pgm", show_height);
}
