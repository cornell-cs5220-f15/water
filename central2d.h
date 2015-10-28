#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <omp.h>
#include <immintrin.h>
#include <iostream>

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
              real cfl = 0.45) :  // Max allowed CFL number
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
        return u_[offset(i+nghost,j+nghost)];
    }
    
    const vec& operator()(int i, int j) const {
        return u_[offset(i+nghost,j+nghost)];
    }

    void teardown();
    
private:
    static constexpr int nghost = 3;   // Number of ghost cells

    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number

    std::vector<vec> u_;            // Solution values
    std::vector<vec> f_;            // Fluxes in x
    std::vector<vec> g_;            // Fluxes in y
    std::vector<vec> ux_;           // x differences of u
    std::vector<vec> uy_;           // y differences of u
    std::vector<vec> fx_;           // x differences of f
    std::vector<vec> gy_;           // y differences of g
    std::vector<vec> v_;            // Solution values at next step

    // Array accessor functions

    int offset(int ix, int iy) const { return iy*nx_all+ix; }

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
    void corrector_float(int x_idx, int y_idx, float dtcdx2, float dtcdy2);

    // Constants
    float *const_0_25   = (float *)_mm_malloc(32, 32);
    float *const_0_0625 = (float *)_mm_malloc(32, 32);
    float *const_neg_1  = (float *)_mm_malloc(32, 32);
    float *const_dtcdx2 = (float *)_mm_malloc(32, 32);
    float *const_dtcdy2 = (float *)_mm_malloc(32, 32);

    // We need to fetch (BLOCK + 1)^2 floats for this to work:
    // So need array of size 81. 81 x sizeof(float) = 81 x 4 = 324
    float *us_0  = (float *)_mm_malloc(324, 32); float *us_1  = (float *)_mm_malloc(324, 32); float *us_2  = (float *)_mm_malloc(324, 32);
    float *fs_0  = (float *)_mm_malloc(324, 32); float *fs_1  = (float *)_mm_malloc(324, 32); float *fs_2  = (float *)_mm_malloc(324, 32);
    float *gs_0  = (float *)_mm_malloc(324, 32); float *gs_1  = (float *)_mm_malloc(324, 32); float *gs_2  = (float *)_mm_malloc(324, 32);
    float *uxs_0 = (float *)_mm_malloc(324, 32); float *uxs_1 = (float *)_mm_malloc(324, 32); float *uxs_2 = (float *)_mm_malloc(324, 32);
    float *uys_0 = (float *)_mm_malloc(324, 32); float *uys_1 = (float *)_mm_malloc(324, 32); float *uys_2 = (float *)_mm_malloc(324, 32);

    // Result array: only do 8x8 = 64. 64 x sizeof(float) = 4
    float *vs_0  = (float *)_mm_malloc(256, 32); float *vs_1 = (float *)_mm_malloc(256, 32); float *vs_2 = (float *)_mm_malloc(256, 32);
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
void Central2D<Physics, Limiter>::teardown()
{
    // Don't forget to free
    _mm_free(const_0_25);
    _mm_free(const_0_0625);
    _mm_free(const_neg_1);
    _mm_free(const_dtcdx2);
    _mm_free(const_dtcdy2);

    _mm_free(us_0);  _mm_free(us_1) ; _mm_free(us_2) ;
    _mm_free(fs_0);  _mm_free(fs_1) ; _mm_free(fs_2) ;
    _mm_free(gs_0);  _mm_free(gs_1) ; _mm_free(gs_2) ;
    _mm_free(uxs_0); _mm_free(uxs_1); _mm_free(uxs_2);
    _mm_free(uys_0); _mm_free(uys_1); _mm_free(uys_2);
    _mm_free(vs_0);  _mm_free(vs_1);  _mm_free(vs_2) ;
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
    using namespace std;
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
    int iy,ix;
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

#define BLOCK 8

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::corrector_float(int x_idx, int y_idx, float dtcdx2, float dtcdy2)
{
    // Populate: copy data over.
    size_t x = x_idx;
    size_t y = y_idx;
    size_t idx = 0;
    #pragma unroll
    for(size_t j = 0; j < BLOCK + 1; ++j) {
        for(size_t i = 0; i < BLOCK + 1; ++i) {
            us_0[idx] = u(x, y)[0]; us_1[idx] = u(x, y)[1]; us_2[idx] = u(x, y)[2]; 
            fs_0[idx] = f(x, y)[0]; fs_1[idx] = f(x, y)[1]; fs_2[idx] = f(x, y)[2]; 
            gs_0[idx] = g(x, y)[0]; gs_1[idx] = g(x, y)[1]; gs_2[idx] = g(x, y)[2]; 
            uxs_0[idx] = ux(x, y)[0]; uxs_1[idx] = ux(x, y)[1]; uxs_2[idx] = ux(x, y)[2]; 
            uys_0[idx] = uy(x, y)[0]; uys_1[idx] = uy(x, y)[1]; uys_2[idx] = uy(x, y)[2]; 
            ++idx; ++x;
        }
        x = x_idx;
        ++y;
    }

    #pragma unroll
    for(size_t i = 0; i < BLOCK; ++i) {
        const_0_25[i]   = 0.25;
        const_0_0625[i] = 0.0625;
        const_neg_1[i]  = -1.0;
        const_dtcdx2[i] = dtcdx2;
        const_dtcdy2[i] = dtcdy2;
    }

    // Declare our 16 registers
    __m256  ymm0, ymm1, ymm2,  ymm3,  ymm4,  ymm5,  ymm6,  ymm7, 
            ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;
    
    // broadcast each of the constants
    ymm12 = _mm256_broadcast_ss(const_0_25); ymm13 = _mm256_broadcast_ss(const_0_0625);
    ymm14 = _mm256_broadcast_ss(const_neg_1); 

    // 0 index first
    {
        ymm0  = _mm256_load_ps((float *) (us_0));  ymm1  = _mm256_load_ps((float *) (us_0 + 1));
        ymm4  = _mm256_load_ps((float *) (uxs_0)); ymm5  = _mm256_load_ps((float *) (uxs_0 + 1));
        ymm8  = _mm256_load_ps((float *) (uys_0)); ymm9  = _mm256_load_ps((float *) (uys_0 + 1));

        ymm2  = _mm256_load_ps((float *) (us_0 + (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_0 + 1 + (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_0 + (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_0 + 1 + (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_0 + (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_0 + 1 + (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);
        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm0
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_0));       ymm4 = _mm256_load_ps((float *) (fs_0 + 1));
        ymm5 = _mm256_load_ps((float *) (fs_0 + (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_0 + 1 + (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_0));       ymm4 = _mm256_load_ps((float *) (gs_0 + 1));
        ymm5 = _mm256_load_ps((float *) (gs_0 + (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_0 + 1 + (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_0), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_0 + (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_0 + 1 + (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_0 + (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_0 + 1 + (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_0 + (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_0 + 1 + (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_0 + 2 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_0 + 1 + 2 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_0 + 2 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_0 + 1 + 2 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_0 + 2 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_0 + 1 + 2 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_0 + (BLOCK + 1))); ymm4 = _mm256_load_ps((float *) (fs_0 + 1 + (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_0 + 2 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_0 + 1 + 2 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_0 + (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_0 + 1 + (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_0 + 2 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_0 + 1 + 2 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_0 + (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_0 + 2 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_0 + 1 + 2 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_0 + 2 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_0 + 1 + 2 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_0 + 2 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_0 + 1 + 2 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_0 + 3 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_0 + 1 + 3 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_0 + 3 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_0 + 1 + 3 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_0 + 3 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_0 + 1 + 3 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_0 + 2 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_0 + 1 + 2 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_0 + 3 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_0 + 1 + 3 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_0 + 2 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_0 + 1 + 2 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_0 + 3 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_0 + 1 + 3 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_0 + 2 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_0 + 3 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_0 + 1 + 3 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_0 + 3 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_0 + 1 + 3 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_0 + 3 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_0 + 1 + 3 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_0 + 4 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_0 + 1 + 4 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_0 + 4 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_0 + 1 + 4 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_0 + 4 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_0 + 1 + 4 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_0 + 3 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_0 + 1 + 3 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_0 + 4 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_0 + 1 + 4 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_0 + 3 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_0 + 1 + 3 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_0 + 4 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_0 + 1 + 4 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_0 + 3 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_0 + 4 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_0 + 1 + 4 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_0 + 4 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_0 + 1 + 4 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_0 + 4 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_0 + 1 + 4 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_0 + 5 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_0 + 1 + 5 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_0 + 5 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_0 + 1 + 5 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_0 + 5 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_0 + 1 + 5 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_0 + 4 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_0 + 1 + 4 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_0 + 5 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_0 + 1 + 5 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_0 + 4 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_0 + 1 + 4 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_0 + 5 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_0 + 1 + 5 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_0 + 4 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_0 + 5 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_0 + 1 + 5 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_0 + 5 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_0 + 1 + 5 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_0 + 5 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_0 + 1 + 5 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_0 + 6 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_0 + 1 + 6 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_0 + 6 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_0 + 1 + 6 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_0 + 6 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_0 + 1 + 6 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // -uy(ix+1, iy+1)[0] + uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_0 + 5 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_0 + 1 + 5 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_0 + 6 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_0 + 1 + 6 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_0 + 5 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_0 + 1 + 5 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_0 + 6 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_0 + 1 + 6 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_0 + 5 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_0 + 6 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_0 + 1 + 6 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_0 + 6 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_0 + 1 + 6 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_0 + 6 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_0 + 1 + 6 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_0 + 7 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_0 + 1 + 7 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_0 + 7 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_0 + 1 + 7 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_0 + 7 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_0 + 1 + 7 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_0 + 6 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_0 + 1 + 6 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_0 + 7 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_0 + 1 + 7 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_0 + 6 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_0 + 1 + 6 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_0 + 7 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_0 + 1 + 7 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_0 + 6 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_0 + 7 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_0 + 1 + 7 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_0 + 7 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_0 + 1 + 7 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_0 + 7 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_0 + 1 + 7 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_0 + 8 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_0 + 1 + 8 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_0 + 8 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_0 + 1 + 8 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_0 + 8 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_0 + 1 + 8 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_0 + 7 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_0 + 1 + 7 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_0 + 8 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_0 + 1 + 8 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_0 + 7 * (BLOCK + 1))); ymm4 = _mm256_load_ps((float *) (gs_0 + 1 + 7 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_0 + 8 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_0 + 1 + 8 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_0 + 7 * (BLOCK)), ymm0);
    }

    // 1 index next
    {
        ymm0  = _mm256_load_ps((float *) (us_1));  ymm1  = _mm256_load_ps((float *) (us_1 + 1));
        ymm4  = _mm256_load_ps((float *) (uxs_1)); ymm5  = _mm256_load_ps((float *) (uxs_1 + 1));
        ymm8  = _mm256_load_ps((float *) (uys_1)); ymm9  = _mm256_load_ps((float *) (uys_1 + 1));

        ymm2  = _mm256_load_ps((float *) (us_1 + (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_1 + 1 + (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_1 + (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_1 + 1 + (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_1 + (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_1 + 1 + (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_1));       ymm4 = _mm256_load_ps((float *) (fs_1 + 1));
        ymm5 = _mm256_load_ps((float *) (fs_1 + (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_1 + 1 + (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_1));       ymm4 = _mm256_load_ps((float *) (gs_1 + 1));
        ymm5 = _mm256_load_ps((float *) (gs_1 + (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_1 + 1 + (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_1), ymm0);
    }


    {
        ymm0  = _mm256_load_ps((float *) (us_1 + (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_1 + 1 + (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_1 + (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_1 + 1 + (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_1 + (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_1 + 1 + (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_1 + 2 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_1 + 1 + 2 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_1 + 2 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_1 + 1 + 2 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_1 + 2 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_1 + 1 + 2 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_1 + (BLOCK + 1))); ymm4 = _mm256_load_ps((float *) (fs_1 + 1 + (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_1 + 2 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_1 + 1 + 2 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_1 + (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_1 + 1 + (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_1 + 2 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_1 + 1 + 2 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_1 + (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_1 + 2 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_1 + 1 + 2 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_1 + 2 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_1 + 1 + 2 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_1 + 2 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_1 + 1 + 2 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_1 + 3 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_1 + 1 + 3 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_1 + 3 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_1 + 1 + 3 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_1 + 3 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_1 + 1 + 3 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_1 + 2 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_1 + 1 + 2 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_1 + 3 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_1 + 1 + 3 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_1 + 2 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_1 + 1 + 2 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_1 + 3 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_1 + 1 + 3 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_1 + 2 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_1 + 3 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_1 + 1 + 3 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_1 + 3 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_1 + 1 + 3 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_1 + 3 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_1 + 1 + 3 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_1 + 4 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_1 + 1 + 4 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_1 + 4 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_1 + 1 + 4 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_1 + 4 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_1 + 1 + 4 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_1 + 3 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_1 + 1 + 3 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_1 + 4 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_1 + 1 + 4 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_1 + 3 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_1 + 1 + 3 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_1 + 4 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_1 + 1 + 4 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_1 + 3 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_1 + 4 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_1 + 1 + 4 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_1 + 4 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_1 + 1 + 4 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_1 + 4 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_1 + 1 + 4 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_1 + 5 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_1 + 1 + 5 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_1 + 5 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_1 + 1 + 5 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_1 + 5 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_1 + 1 + 5 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_1 + 4 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_1 + 1 + 4 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_1 + 5 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_1 + 1 + 5 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_1 + 4 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_1 + 1 + 4 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_1 + 5 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_1 + 1 + 5 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_1 + 4 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_1 + 5 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_1 + 1 + 5 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_1 + 5 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_1 + 1 + 5 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_1 + 5 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_1 + 1 + 5 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_1 + 6 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_1 + 1 + 6 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_1 + 6 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_1 + 1 + 6 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_1 + 6 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_1 + 1 + 6 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_1 + 5 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_1 + 1 + 5 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_1 + 6 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_1 + 1 + 6 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_1 + 5 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_1 + 1 + 5 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_1 + 6 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_1 + 1 + 6 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_1 + 5 * (BLOCK)), ymm0);
    }



    {
        ymm0  = _mm256_load_ps((float *) (us_1 + 6 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_1 + 1 + 6 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_1 + 6 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_1 + 1 + 6 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_1 + 6 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_1 + 1 + 6 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_1 + 7 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_1 + 1 + 7 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_1 + 7 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_1 + 1 + 7 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_1 + 7 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_1 + 1 + 7 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_1 + 6 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_1 + 1 + 6 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_1 + 7 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_1 + 1 + 7 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_1 + 6 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_1 + 1 + 6 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_1 + 7 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_1 + 1 + 7 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_1 + 6 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_1 + 7 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_1 + 1 + 7 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_1 + 7 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_1 + 1 + 7 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_1 + 7 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_1 + 1 + 7 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_1 + 8 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_1 + 1 + 8 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_1 + 8 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_1 + 1 + 8 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_1 + 8 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_1 + 1 + 8 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_1 + 7 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_1 + 1 + 7 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_1 + 8 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_1 + 1 + 8 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_1 + 7 * (BLOCK + 1))); ymm4 = _mm256_load_ps((float *) (gs_1 + 1 + 7 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_1 + 8 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_1 + 1 + 8 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_1 + 7 * (BLOCK)), ymm0);
    }

    // Finally, 2 index
    {
        ymm0  = _mm256_load_ps((float *) (us_2));  ymm1  = _mm256_load_ps((float *) (us_2 + 1));
        ymm4  = _mm256_load_ps((float *) (uxs_2)); ymm5  = _mm256_load_ps((float *) (uxs_2 + 1));
        ymm8  = _mm256_load_ps((float *) (uys_2)); ymm9  = _mm256_load_ps((float *) (uys_2 + 1));

        ymm2  = _mm256_load_ps((float *) (us_2 + (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_2 + 1 + (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_2 + (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_2 + 1 + (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_2 + (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_2 + 1 + (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_2));       ymm4 = _mm256_load_ps((float *) (fs_2 + 1));
        ymm5 = _mm256_load_ps((float *) (fs_2 + (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_2 + 1 + (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_2));       ymm4 = _mm256_load_ps((float *) (gs_2 + 1));
        ymm5 = _mm256_load_ps((float *) (gs_2 + (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_2 + 1 + (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_2), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_2 + (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_2 + 1 + (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_2 + (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_2 + 1 + (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_2 + (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_2 + 1 + (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_2 + 2 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_2 + 1 + 2 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_2 + 2 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_2 + 1 + 2 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_2 + 2 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_2 + 1 + 2 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_2 + (BLOCK + 1))); ymm4 = _mm256_load_ps((float *) (fs_2 + 1 + (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_2 + 2 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_2 + 1 + 2 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_2 + (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_2 + 1 + (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_2 + 2 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_2 + 1 + 2 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_2 + (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_2 + 2 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_2 + 1 + 2 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_2 + 2 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_2 + 1 + 2 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_2 + 2 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_2 + 1 + 2 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_2 + 3 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_2 + 1 + 3 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_2 + 3 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_2 + 1 + 3 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_2 + 3 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_2 + 1 + 3 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_2 + 2 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_2 + 1 + 2 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_2 + 3 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_2 + 1 + 3 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_2 + 2 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_2 + 1 + 2 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_2 + 3 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_2 + 1 + 3 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_2 + 2 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_2 + 3 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_2 + 1 + 3 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_2 + 3 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_2 + 1 + 3 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_2 + 3 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_2 + 1 + 3 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_2 + 4 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_2 + 1 + 4 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_2 + 4 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_2 + 1 + 4 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_2 + 4 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_2 + 1 + 4 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_2 + 3 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_2 + 1 + 3 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_2 + 4 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_2 + 1 + 4 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_2 + 3 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_2 + 1 + 3 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_2 + 4 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_2 + 1 + 4 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_2 + 3 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_2 + 4 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_2 + 1 + 4 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_2 + 4 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_2 + 1 + 4 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_2 + 4 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_2 + 1 + 4 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_2 + 5 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_2 + 1 + 5 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_2 + 5 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_2 + 1 + 5 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_2 + 5 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_2 + 1 + 5 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_2 + 4 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_2 + 1 + 4 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_2 + 5 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_2 + 1 + 5 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_2 + 4 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_2 + 1 + 4 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_2 + 5 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_2 + 1 + 5 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_2 + 4 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_2 + 5 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_2 + 1 + 5 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_2 + 5 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_2 + 1 + 5 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_2 + 5 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_2 + 1 + 5 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_2 + 6 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_2 + 1 + 6 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_2 + 6 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_2 + 1 + 6 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_2 + 6 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_2 + 1 + 6 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_2 + 5 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_2 + 1 + 5 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_2 + 6 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_2 + 1 + 6 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_2 + 5 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_2 + 1 + 5 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_2 + 6 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_2 + 1 + 6 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_2 + 5 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_2 + 6 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_2 + 1 + 6 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_2 + 6 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_2 + 1 + 6 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_2 + 6 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_2 + 1 + 6 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_2 + 7 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_2 + 1 + 7 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_2 + 7 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_2 + 1 + 7 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_2 + 7 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_2 + 1 + 7 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_2 + 6 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_2 + 1 + 6 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_2 + 7 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_2 + 1 + 7 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_2 + 6 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (gs_2 + 1 + 6 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_2 + 7 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_2 + 1 + 7 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_2 + 6 * (BLOCK)), ymm0);
    }

    {
        ymm0  = _mm256_load_ps((float *) (us_2 + 7 * (BLOCK + 1)));  ymm1  = _mm256_load_ps((float *) (us_2 + 1 + 7 * (BLOCK + 1)));
        ymm4  = _mm256_load_ps((float *) (uxs_2 + 7 * (BLOCK + 1))); ymm5  = _mm256_load_ps((float *) (uxs_2 + 1 + 7 * (BLOCK + 1)));
        ymm8  = _mm256_load_ps((float *) (uys_2 + 7 * (BLOCK + 1))); ymm9  = _mm256_load_ps((float *) (uys_2 + 1 + 7 * (BLOCK + 1)));

        ymm2  = _mm256_load_ps((float *) (us_2 + 8 * (BLOCK + 1)));  ymm3  = _mm256_load_ps((float *) (us_2 + 1 + 8 * (BLOCK + 1)));
        ymm6  = _mm256_load_ps((float *) (uxs_2 + 8 * (BLOCK + 1))); ymm7  = _mm256_load_ps((float *) (uxs_2 + 1 + 8 * (BLOCK + 1)));
        ymm10 = _mm256_load_ps((float *) (uys_2 + 8 * (BLOCK + 1))); ymm11 = _mm256_load_ps((float *) (uys_2 + 1 + 8 * (BLOCK + 1)));

        // Perform additions first : u -> ymm0
        ymm0 = _mm256_add_ps(ymm0, ymm1); 
        ymm1 = _mm256_add_ps(ymm2, ymm3);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // -ux(ix+1, iy)[0] + ux(ix, iy)[0] -> ymm4
        ymm15 = _mm256_mul_ps(ymm5, ymm14); 
        ymm4  = _mm256_add_ps(ymm4, ymm15);

        // -ux(ix+1, iy+1)[0] + ux(ix, iy+1)[0] -> ymm5
        ymm15 = _mm256_mul_ps(ymm7, ymm14);
        ymm5  = _mm256_add_ps(ymm6, ymm15);

        // -uy(ix, iy+1)[0] + uy(ix, iy)[0] -> ymm8
        ymm15 = _mm256_mul_ps(ymm10, ymm14);
        ymm8  = _mm256_add_ps(ymm8,  ymm15);

        // uy(ix+1, iy+1)[0] - uy(ix+1, iy)[0] -> ymm9
        ymm15 = _mm256_mul_ps(ymm11, ymm14);
        ymm9  = _mm256_add_ps(ymm9,  ymm15);

        // sum ux and uy -> ymm1
        ymm1  = _mm256_add_ps(ymm9, ymm8);
        ymm15 = _mm256_add_ps(ymm4, ymm5);
        ymm1  = _mm256_add_ps(ymm1, ymm15);

        // Add everything together for u, ux, uy -> ymm15
        ymm0 = _mm256_mul_ps(ymm0, ymm12);
        ymm1 = _mm256_mul_ps(ymm13, ymm1);
        ymm0 = _mm256_add_ps(ymm0, ymm1);

        // Need to hold onto ymm0 and can use 1, 4, 5, 8, 9 for f and g.
        // First f
        ymm1 = _mm256_load_ps((float *) (fs_2 + 7 * (BLOCK + 1)));       ymm4 = _mm256_load_ps((float *) (fs_2 + 1 + 7 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (fs_2 + 8 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (fs_2 + 1 + 8 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdx2);

        // -f(ix+1, iy)[0] + f(ix, iy)[0] -> ymm1
        ymm4 = _mm256_mul_ps(ymm4, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm4);

        // -f(ix+1, iy+1)[0] + f(ix, iy+1)[0] -> ymm5
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm5 = _mm256_add_ps(ymm5, ymm8);

        // Sum f and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm5);
        ymm5 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm5);

        // Now g
        ymm1 = _mm256_load_ps((float *) (gs_2 + 7 * (BLOCK + 1))); ymm4 = _mm256_load_ps((float *) (gs_2 + 1 + 7 * (BLOCK + 1)));
        ymm5 = _mm256_load_ps((float *) (gs_2 + 8 * (BLOCK + 1))); ymm8 = _mm256_load_ps((float *) (gs_2 + 1 + 8 * (BLOCK + 1)));
        ymm9 = _mm256_broadcast_ss(const_dtcdy2);

        // -g(ix, iy+1)[0] + g(ix, iy)[0] -> ymm1
        ymm5 = _mm256_mul_ps(ymm5, ymm14);
        ymm1 = _mm256_add_ps(ymm1, ymm5);

        // -g(ix+1, iy+1)[0] + g(ix+1, iy)[0] -> ymm4
        ymm8 = _mm256_mul_ps(ymm8, ymm14);
        ymm4 = _mm256_add_ps(ymm4, ymm8);

        // Sum g and add to ymm0
        ymm1 = _mm256_add_ps(ymm1, ymm4);
        ymm4 = _mm256_mul_ps(ymm1, ymm9);
        ymm0 = _mm256_add_ps(ymm0, ymm4);

        // Aaaand store
        _mm256_store_ps((float *) (vs_2 + 7 * (BLOCK)), ymm0);
    }

    // Now copy v back in
    x = x_idx;
    y = y_idx;
    idx = 0;
    #pragma unroll
    for(size_t j = 0; j < BLOCK; ++j) {
        #pragma unroll
        for(size_t i = 0; i < BLOCK; ++i) {
            v(x, y)[0] = vs_0[idx]; v(x, y)[1] = vs_1[idx]; v(x, y)[2] = vs_2[idx];
            ++idx; ++x;
        }
        ++y; x = x_idx;
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
    int iy,ix,m,i,j;

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
    double t0 = omp_get_wtime();
    int num_blocks_x = nx / BLOCK;
    int num_blocks_y = ny / BLOCK;
    int loop_start = nghost-io;
    int loop_end_x = nx+nghost-io;
    int loop_end_y = ny+nghost-io;

    int num_blocks_remaining_x = nx - num_blocks_x;
    int num_blocks_remaining_y = ny - num_blocks_y;

    int x_idx, y_idx;
    for(int block_y = 0; block_y < num_blocks_y; ++block_y) {
        y_idx = loop_start + block_y * BLOCK;
        for(int block_x = 0; block_x < num_blocks_x; ++block_x) {
            x_idx = loop_start + block_x * BLOCK;
            corrector_float(x_idx, y_idx, dtcdx2, dtcdy2);
        }
        x_idx = loop_start + num_blocks_x * BLOCK;
        for(x_idx; x_idx < loop_end_x; ++x_idx) {
            v(x_idx,y_idx)[0] =
                0.2500 * ( u(x_idx,  y_idx)[0] + u(x_idx+1,y_idx  )[0] +
                           u(x_idx,y_idx+1)[0] + u(x_idx+1,y_idx+1)[0] ) -
                0.0625 * ( ux(x_idx+1,y_idx  )[0] - ux(x_idx,y_idx  )[0] +
                           ux(x_idx+1,y_idx+1)[0] - ux(x_idx,y_idx+1)[0] +
                           uy(x_idx,  y_idx+1)[0] - uy(x_idx,  y_idx)[0] +
                           uy(x_idx+1,y_idx+1)[0] - uy(x_idx+1,y_idx)[0] ) -
                dtcdx2 * ( f(x_idx+1,y_idx  )[0] - f(x_idx,y_idx  )[0] +
                           f(x_idx+1,y_idx+1)[0] - f(x_idx,y_idx+1)[0] ) -
                dtcdy2 * ( g(x_idx,  y_idx+1)[0] - g(x_idx,  y_idx)[0] +
                           g(x_idx+1,y_idx+1)[0] - g(x_idx+1,y_idx)[0] );

            v(x_idx,y_idx)[1] =
                0.2500 * ( u(x_idx,  y_idx)[1] + u(x_idx+1,y_idx  )[1] +
                           u(x_idx,y_idx+1)[1] + u(x_idx+1,y_idx+1)[1] ) -
                0.0625 * ( ux(x_idx+1,y_idx  )[1] - ux(x_idx,y_idx  )[1] +
                           ux(x_idx+1,y_idx+1)[1] - ux(x_idx,y_idx+1)[1] +
                           uy(x_idx,  y_idx+1)[1] - uy(x_idx,  y_idx)[1] +
                           uy(x_idx+1,y_idx+1)[1] - uy(x_idx+1,y_idx)[1] ) -
                dtcdx2 * ( f(x_idx+1,y_idx  )[1] - f(x_idx,y_idx  )[1] +
                           f(x_idx+1,y_idx+1)[1] - f(x_idx,y_idx+1)[1] ) -
                dtcdy2 * ( g(x_idx,  y_idx+1)[1] - g(x_idx,  y_idx)[1] +
                           g(x_idx+1,y_idx+1)[1] - g(x_idx+1,y_idx)[1] );

            v(x_idx,y_idx)[2] =
                0.2500 * ( u(x_idx,  y_idx)[2] + u(x_idx+1,y_idx  )[2] +
                           u(x_idx,y_idx+1)[2] + u(x_idx+1,y_idx+1)[2] ) -
                0.0625 * ( ux(x_idx+1,y_idx  )[2] - ux(x_idx,y_idx  )[2] +
                           ux(x_idx+1,y_idx+1)[2] - ux(x_idx,y_idx+1)[2] +
                           uy(x_idx,  y_idx+1)[2] - uy(x_idx,  y_idx)[2] +
                           uy(x_idx+1,y_idx+1)[2] - uy(x_idx+1,y_idx)[2] ) -
                dtcdx2 * ( f(x_idx+1,y_idx  )[2] - f(x_idx,y_idx  )[2] +
                           f(x_idx+1,y_idx+1)[2] - f(x_idx,y_idx+1)[2] ) -
                dtcdy2 * ( g(x_idx,  y_idx+1)[2] - g(x_idx,  y_idx)[2] +
                           g(x_idx+1,y_idx+1)[2] - g(x_idx+1,y_idx)[2] );
        }
    }

    y_idx = loop_start + num_blocks_y * BLOCK;
    for(y_idx; y_idx < loop_end_y; ++y_idx) {
        x_idx = loop_start;
        for(x_idx; x_idx < loop_end_x; ++x_idx) {
            v(x_idx,y_idx)[0] =
                0.2500 * ( u(x_idx,  y_idx)[0] + u(x_idx+1,y_idx  )[0] +
                           u(x_idx,y_idx+1)[0] + u(x_idx+1,y_idx+1)[0] ) -
                0.0625 * ( ux(x_idx+1,y_idx  )[0] - ux(x_idx,y_idx  )[0] +
                           ux(x_idx+1,y_idx+1)[0] - ux(x_idx,y_idx+1)[0] +
                           uy(x_idx,  y_idx+1)[0] - uy(x_idx,  y_idx)[0] +
                           uy(x_idx+1,y_idx+1)[0] - uy(x_idx+1,y_idx)[0] ) -
                dtcdx2 * ( f(x_idx+1,y_idx  )[0] - f(x_idx,y_idx  )[0] +
                           f(x_idx+1,y_idx+1)[0] - f(x_idx,y_idx+1)[0] ) -
                dtcdy2 * ( g(x_idx,  y_idx+1)[0] - g(x_idx,  y_idx)[0] +
                           g(x_idx+1,y_idx+1)[0] - g(x_idx+1,y_idx)[0] );

            v(x_idx,y_idx)[1] =
                0.2500 * ( u(x_idx,  y_idx)[1] + u(x_idx+1,y_idx  )[1] +
                           u(x_idx,y_idx+1)[1] + u(x_idx+1,y_idx+1)[1] ) -
                0.0625 * ( ux(x_idx+1,y_idx  )[1] - ux(x_idx,y_idx  )[1] +
                           ux(x_idx+1,y_idx+1)[1] - ux(x_idx,y_idx+1)[1] +
                           uy(x_idx,  y_idx+1)[1] - uy(x_idx,  y_idx)[1] +
                           uy(x_idx+1,y_idx+1)[1] - uy(x_idx+1,y_idx)[1] ) -
                dtcdx2 * ( f(x_idx+1,y_idx  )[1] - f(x_idx,y_idx  )[1] +
                           f(x_idx+1,y_idx+1)[1] - f(x_idx,y_idx+1)[1] ) -
                dtcdy2 * ( g(x_idx,  y_idx+1)[1] - g(x_idx,  y_idx)[1] +
                           g(x_idx+1,y_idx+1)[1] - g(x_idx+1,y_idx)[1] );

            v(x_idx,y_idx)[2] =
                0.2500 * ( u(x_idx,  y_idx)[2] + u(x_idx+1,y_idx  )[2] +
                           u(x_idx,y_idx+1)[2] + u(x_idx+1,y_idx+1)[2] ) -
                0.0625 * ( ux(x_idx+1,y_idx  )[2] - ux(x_idx,y_idx  )[2] +
                           ux(x_idx+1,y_idx+1)[2] - ux(x_idx,y_idx+1)[2] +
                           uy(x_idx,  y_idx+1)[2] - uy(x_idx,  y_idx)[2] +
                           uy(x_idx+1,y_idx+1)[2] - uy(x_idx+1,y_idx)[2] ) -
                dtcdx2 * ( f(x_idx+1,y_idx  )[2] - f(x_idx,y_idx  )[2] +
                           f(x_idx+1,y_idx+1)[2] - f(x_idx,y_idx+1)[2] ) -
                dtcdy2 * ( g(x_idx,  y_idx+1)[2] - g(x_idx,  y_idx)[2] +
                           g(x_idx+1,y_idx+1)[2] - g(x_idx+1,y_idx)[2] );
        }
    }

    // int loop_start = nghost-io;
    // int loop_end_x = nx+nghost-io;
    // int loop_end_y = ny+nghost-io;
    // for (int iy = loop_start; iy < loop_end_y; ++iy)
    //     for (int ix = loop_start; ix < loop_end_x; ++ix) {
    //             v(ix,iy)[0] =
    //                 0.2500 * ( u(ix,  iy)[0] + u(ix+1,iy  )[0] +
    //                            u(ix,iy+1)[0] + u(ix+1,iy+1)[0] ) -
    //                 0.0625 * ( ux(ix+1,iy  )[0] - ux(ix,iy  )[0] +
    //                            ux(ix+1,iy+1)[0] - ux(ix,iy+1)[0] +
    //                            uy(ix,  iy+1)[0] - uy(ix,  iy)[0] +
    //                            uy(ix+1,iy+1)[0] - uy(ix+1,iy)[0] ) -
    //                 dtcdx2 * ( f(ix+1,iy  )[0] - f(ix,iy  )[0] +
    //                            f(ix+1,iy+1)[0] - f(ix,iy+1)[0] ) -
    //                 dtcdy2 * ( g(ix,  iy+1)[0] - g(ix,  iy)[0] +
    //                            g(ix+1,iy+1)[0] - g(ix+1,iy)[0] );
    //             v(ix,iy)[1] =
    //                 0.2500 * ( u(ix,  iy)[1] + u(ix+1,iy  )[1] +
    //                            u(ix,iy+1)[1] + u(ix+1,iy+1)[1] ) -
    //                 0.0625 * ( ux(ix+1,iy  )[1] - ux(ix,iy  )[1] +
    //                            ux(ix+1,iy+1)[1] - ux(ix,iy+1)[1] +
    //                            uy(ix,  iy+1)[1] - uy(ix,  iy)[1] +
    //                            uy(ix+1,iy+1)[1] - uy(ix+1,iy)[1] ) -
    //                 dtcdx2 * ( f(ix+1,iy  )[1] - f(ix,iy  )[1] +
    //                            f(ix+1,iy+1)[1] - f(ix,iy+1)[1] ) -
    //                 dtcdy2 * ( g(ix,  iy+1)[1] - g(ix,  iy)[1] +
    //                            g(ix+1,iy+1)[1] - g(ix+1,iy)[1] );

    //             v(ix,iy)[2] =
    //                 0.2500 * ( u(ix,  iy)[2] + u(ix+1,iy  )[2] +
    //                            u(ix,iy+1)[2] + u(ix+1,iy+1)[2] ) -
    //                 0.0625 * ( ux(ix+1,iy  )[2] - ux(ix,iy  )[2] +
    //                            ux(ix+1,iy+1)[2] - ux(ix,iy+1)[2] +
    //                            uy(ix,  iy+1)[2] - uy(ix,  iy)[2] +
    //                            uy(ix+1,iy+1)[2] - uy(ix+1,iy)[2] ) -
    //                 dtcdx2 * ( f(ix+1,iy  )[2] - f(ix,iy  )[2] +
    //                            f(ix+1,iy+1)[2] - f(ix,iy+1)[2] ) -
    //                 dtcdy2 * ( g(ix,  iy+1)[2] - g(ix,  iy)[2] +
    //                            g(ix+1,iy+1)[2] - g(ix+1,iy)[2] );
    //     }

    // for(int iy = loop_start; iy < loop_start + 8; ++iy) {
    //     for(int ix = loop_start; ix < loop_start + 8; ++ix) {
    //         std::cout << "v = " << v(ix, iy)[0] << ", " << v(ix, iy)[1] << ", " << v(ix, iy)[2] << std::endl;
    //     }
    // }

    // Copy from v storage back to main grid
    for (int j = nghost; j < ny+nghost; ++j){
        for (int i = nghost; i < nx+nghost; ++i){
            u(i,j) = v(i-io,j-io);
        }
    }
    double t1 = omp_get_wtime();
    printf("Time: %e\n", t1-t0);
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
