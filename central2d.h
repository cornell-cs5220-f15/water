#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

#include <omp.h>

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
              int nodomains,      //No of domains of the board
              real cfl = 0.45) :  // Max allowed CFL number
    nodomains(nodomains),
    nx(nx), ny(ny),
    domain_nx((int) (ceil((real) nx / nodomains))),
    domain_ny((int) (ceil((real) ny / nodomains))),
    nx_all(nodomains * (domain_nx + 2 * nghost)),
    ny_all(nodomains * (domain_ny + 2 * nghost)),
    domain_nx_inc_ghost( domain_nx + 2 * nghost),
    domain_ny_inc_ghost( domain_ny + 2 * nghost),
    dx(w/nx), dy(h/ny),
    cfl(cfl), 
    u_ (nx_all * ny_all),
    f_ (nx_all * ny_all),
    g_ (nx_all * ny_all),
    ux_(nx_all * ny_all),
    uy_(nx_all * ny_all),
    fx_(nx_all * ny_all),
    gy_(nx_all * ny_all),
    v_ (nx_all * ny_all),
    ures_ (nx * ny) {}

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
        return u_[offsetfull(i+nghost,j+nghost)]; //TODO
    }
    
    const vec& operator()(int i, int j) const {
        return u_[offsetfull(i+nghost,j+nghost)]; //TODO
    }
    
private:
    static constexpr int nghost = 3;   // Number of ghost cells

    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const int nodomains;
    const int domain_nx, domain_ny;
    const int domain_nx_inc_ghost, domain_ny_inc_ghost;
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
    std::vector<vec> ures_;         // Joined solution values in one compressed board.

    // Array accessor functions

    // access the (ix, iy) element of the tno'th subdomain, ix and iy are relative 
    // indices 0 <= ix, iy < domain_nx, domain_ny
    int offset(int ix, int iy, int tno) const { 
        return (nghost+iy+(tno/nodomains)*domain_ny_inc_ghost)*nx_all + (tno%nodomains)*domain_nx_inc_ghost + nghost + ix; 
    }
    
    // access the (ix, iy) element of the tno'th subdomain, ix and iy are absolute  
    // indices 0 <= ix, iy < nx, ny
    int offsetabs(int ix, int iy, int tno) const {
        return (nghost+iy+2*nghost*(iy/domain_ny))*nx_all + nghost+ix+2*nghost*(ix/domain_nx);
    }
    
    // access the (ix, iy) element of the whole nx_all-by-ny_all board. If the coordinate
    // is outside the board, move it in using the periodic re-enter strategy.
    int offsetfull (int ix, int iy) const {
        ix = (ix + nx_all) % nx_all;
        iy = (iy + ny_all) % ny_all;
        return iy*nx_all + ix;
    }
    
    // access the (ix, iy) element of the tno'th subdomain, ix and iy are relative 
    // indices 0 <= ix, iy < domain_nx_inc_ghost, domain_ny_inc_ghost
    int offsetg (int ix, int iy, int tno) const {
        return (iy+(tno/nodomains)*domain_ny_inc_ghost)*nx_all + (tno%nodomains)*domain_nx_inc_ghost + ix;
    }

    vec& u(int ix, int iy, int tno)    { return u_[offset(ix, iy, tno)]; }
    vec& uf(int ix, int iy)            { return u_[offsetfull(ix, iy)];  }
    vec& ug(int ix, int iy, int tno)   { return u_[offsetg(ix, iy, tno)];}
    vec& v(int ix, int iy, int tno)    { return v_[offsetg(ix, iy, tno)]; }
    vec& f(int ix, int iy, int tno)    { return f_[offsetg(ix, iy, tno)];}
    vec& g(int ix, int iy, int tno)    { return g_[offsetg(ix, iy, tno)];}

    vec& ux(int ix, int iy, int tno)   { return ux_[offsetg(ix, iy, tno)]; }
    vec& uy(int ix, int iy, int tno)   { return uy_[offsetg(ix, iy, tno)]; }
    vec& fx(int ix, int iy, int tno)   { return fx_[offsetg(ix, iy, tno)]; }
    vec& gy(int ix, int iy, int tno)   { return gy_[offsetg(ix, iy, tno)]; }

    // Wrapped accessor (periodic BC)
    //TODO: Need to change this
    int ioffset(int ix, int iy) {
        return offset( (ix+nx-nghost) % nx + nghost,
           (iy+ny-nghost) % ny + nghost );
    }

    //TODO: Remove this because this is not required
    vec& uwrap(int ix, int iy)  { return u_[ioffset(ix,iy)]; }

    // Apply limiter to all components in a vector
    static void limdiff(vec& du, const vec& um, const vec& u0, const vec& up) {
        for (int m = 0; m < du.size(); ++m)
            du[m] = Limiter::limdiff(um[m], u0[m], up[m]);
    }

    // Stages of the main algorithm
    void apply_periodic();
    void compute_fg_speeds(real& cx, real& cy, int tno);
    void limited_derivs(int tno);
    void compute_step(int io, real dt, int tno);

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
//TODO: Even this can be parallelized
template <class Physics, class Limiter>
template <typename F>
 void Central2D<Physics, Limiter>::init(F f)
 {
    for (int tno=0; tno < nodomains*nodomains; tno++)
        for (int iy = 0; iy < ny; ++iy)
            for (int ix = 0; ix < nx; ++ix)
                f(u(ix,iy,tno), (ix+0.5)*dx, (iy+0.5)*dy);
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
        
        for (int i=0; i<nodomains; i++)
            for (int j=0; j<ny_all; j++)
                for (int k=0; k<nghost; k++) {
                    uf(k+i*domain_nx_inc_ghost, j) = uf(k+i*domain_nx_inc_ghost-2*nghost, j);
                    uf(k+i*domain_nx_inc_ghost+nghost+domain_nx, j) = uf(k+i*domain_nx_inc_ghost+3*nghost+domain_nx, j);
                }
                
        for (int i=0; i<nodomains; i++)
            for (int j=0; j<nx_all; j++)
                for (int k=0; k<nghost; k++) {
                    uf(j, k+i*domain_ny_inc_ghost) = uf(j, k+i*domain_ny_inc_ghost -2*nghost);
                    uf(j, k+i*domain_ny_inc_ghost+nghost+domain_ny) = uf(j, k+i*domain_ny_inc_ghost+3*nghost+domain_ny);
                }
// Copy data between right and left boundaries
//     /*for (int iy = 0; iy < ny_all; ++iy)
//         for (int ix = 0; ix < nghost; ++ix) {
//             u(ix,          iy) = uwrap(ix,          iy);
//             u(nx+nghost+ix,iy) = uwrap(nx+nghost+ix,iy);
//         }
//    
// 
//         int x_limit = (tno % nodomains == nodomains - 1) ? nodomains * domain_nx - nx : 0;
//         int y_limit = (tno / nodomains == nodomains - 1) ? nodomains * domain_ny - ny : 0;
// 
//         for (int y = 0; iy < domain_ny_inc_ghost; y++){
//             for (int x = 0; x < nghost; x++){
//                 u(x, y, tno) = board((domain_nx * (tno % nodomains) - nghost + nx + x) % nx, (domain_ny * (tno/nodomains) - nghost + ny + y) % ny);
//             }
// 
//             for (int x = nghost + domain_nx - x_limit; x < nghost * 2 + domain_nx; x++){
//                 u(x, y, tno) = board((domain_nx * (tno % nodomains) - nghost + nx + x) % nx, (domain_ny * (tno / nodomains) - nghost + ny + y) % ny);
//             }
//         }
// 
// 
//     /*for (int ix = 0; ix < nx_all; ++ix)
//         for (int iy = 0; iy < nghost; ++iy) {
//             u(ix,          iy) = uwrap(ix,          iy);
//             u(ix,ny+nghost+iy) = uwrap(ix,ny+nghost+iy);
//         }
//     }*/
// 
//     // Copy data between top and bottom boundari
//     for (int x = 0; x < domain_nx_inc_ghost; x++){
//         for (int y = 0; y < nghost; y++){
//             u(x, y, tno) = board((domain_nx * (tno % nodomains) - nghost + nx + x) % nx, (domain_ny * (tno / nodomains) - nghost + ny + y) % ny);
//         }
// 
//         for (int y = nghost + domain_ny - y_limit; y < nghost * 2 + domain_ny; y++){
//             u(x, y, tno) = board((domain_nx * (tno % nodomains) - nghost + nx + x) % nx, (domain_ny * (tno / nodomains) - nghost + ny + y) % ny);
//         }
//     }
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
void Central2D<Physics, Limiter>::compute_fg_speeds(real& cx_, real& cy_, int tno)
{
    using namespace std;
    real cx = 1.0e-15;
    real cy = 1.0e-15;
    for (int iy = 0; iy < domain_ny_inc_ghost; ++iy)
        for (int ix = 0; ix < domain_nx_inc_ghost; ++ix) {
            real cell_cx, cell_cy;
            Physics::flux(f(ix,iy, tno), g(ix,iy, tno), ug(ix,iy, tno));
            Physics::wave_speed(cell_cx, cell_cy, ug(ix,iy, tno));
            cx = cx > cell_cx ? cx : cell_cx;
            cy = cy > cell_cy ? cy : cell_cy;
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
 void Central2D<Physics, Limiter>::limited_derivs(int tno)
 {
    int iy, ix;
    for (iy = 1; iy < domain_ny_inc_ghost-1; ++iy) {
        for (ix = 1; ix < domain_nx_inc_ghost-1; ++ix) {

        // x derivs
            limdiff( ux(ix,iy,tno), ug(ix-1,iy,tno), ug(ix,iy,tno), ug(ix+1,iy,tno) );
            limdiff( fx(ix,iy,tno), f(ix-1,iy,tno), f(ix,iy,tno), f(ix+1,iy,tno) );

        // y derivs
            limdiff( uy(ix,iy,tno), ug(ix,iy-1,tno), ug(ix,iy,tno), ug(ix,iy+1,tno) );
            limdiff( gy(ix,iy,tno), g(ix,iy-1,tno), g(ix,iy,tno), g(ix,iy+1,tno) );
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
 void Central2D<Physics, Limiter>::compute_step(int io, real dt, int tno)
 {
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;

    // Predictor (flux values of f and g at half step)
    for (int iy = 1; iy < domain_ny_inc_ghost-1; ++iy)
        for (int ix = 1; ix < domain_nx_inc_ghost-1; ++ix) {
            vec uh = ug(ix,iy,tno);
            for (int m = 0; m < uh.size(); ++m) {
                uh[m] -= dtcdx2 * fx(ix,iy,tno)[m];
                uh[m] -= dtcdy2 * gy(ix,iy,tno)[m];
            }
            Physics::flux(f(ix,iy,tno), g(ix,iy,tno), uh);
        }

    // Corrector (finish the step)
        for (int iy = nghost-io; iy < domain_ny+nghost-io; ++iy)
            for (int ix = nghost-io; ix < domain_nx+nghost-io; ++ix) {
                for (int m = 0; m < v(ix,iy,tno).size(); ++m) {
                    v(ix,iy,tno)[m] =
                    0.2500 * ( ug(ix,iy,tno)[m] + ug(ix+1,iy,tno)[m] +
                       ug(ix,iy+1,tno)[m] + ug(ix+1,iy+1,tno)[m] ) -
                    0.0625 * ( ux(ix+1,iy,tno)[m] - ux(ix,iy,tno)[m] +
                       ux(ix+1,iy+1,tno)[m] - ux(ix,iy+1,tno)[m] +
                       uy(ix,  iy+1,tno)[m] - uy(ix,  iy,tno)[m] +
                       uy(ix+1,iy+1,tno)[m] - uy(ix+1,iy,tno)[m] ) -
                    dtcdx2 * ( f(ix+1,iy,tno)[m] - f(ix,iy,tno)[m] +
                       f(ix+1,iy+1,tno)[m] - f(ix,iy+1,tno)[m] ) -
                    dtcdy2 * ( g(ix,  iy+1,tno)[m] - g(ix,  iy,tno)[m] +
                       g(ix+1,iy+1,tno)[m] - g(ix+1,iy,tno)[m] );
                }
            }

    // Copy from v storage back to main grid
            for (int j = nghost; j < domain_ny+nghost; ++j){
                for (int i = nghost; i < domain_nx+nghost; ++i){
                    ug(i,j,tno) = v(i-io,j-io,tno);
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
    real dt;
    real cx[nodomains*nodomains], cy[nodomains*nodomains];
    #pragma omp parallel num_threads(nodomains*nodomains)
    while (!done) {        
        int tno = omp_get_thread_num();        
        #pragma omp single
        apply_periodic();
        compute_fg_speeds(cx[tno], cy[tno], tno);
        limited_derivs(tno);
        #pragma omp barrier
        {
            #pragma omp single
            {
                real cxmax=cx[0], cymax=cy[0];
                for (int i=1; i<nodomains*nodomains; i++) {
                    cxmax = (cxmax>cx[i])?cxmax:cx[i];
                    cymax = (cymax>cy[i])?cymax:cy[i];
                }
          
                dt = cfl / (cxmax/dx > cymax/dy ? cxmax/dx : cymax/dy);
        
                if (t + 2*dt >= tfinal) {
                    dt = (tfinal-t)/2;
                    done = true;
                }
            }
        }        
        compute_step(0, dt, tno);
        compute_step(1, dt, tno);
        #pragma omp barrier
        #pragma omp single
        t += 2*dt;
    }
        
//         for (int io = 0; io < 2; ++io) {
//             real cx[] = real[nodomains*nodomains], cy[]= real[nodomains*nodomains];
//             #paragma omp single
//             apply_periodic();
//             compute_fg_speeds(cx+tno, cy+tno, tno); 
//             limited_derivs(tno);
//             if (io == 0) {
//                 #pragma omp barrier 
//                 {
//                     #pragma omp single
//                     {
//                         real cxmax=cx[0], cymax=cy[0];
//                         for (int i=1; i<nodomains*nodomains; i++) {
//                             cxmax = (cxmax>cx[i])?cxmax:cx[i];
//                             cymax = (cymax>cy[i])?cymax:cy[i];
//                         }
//                       
//                         dt = cfl / (cxmax/dx > cymax/dy ? cxmax/dx : cymax/dy);
//                     
//                         if (t + 2*dt >= tfinal) {
//                             dt = (tfinal-t)/2;
//                             done = true;
//                         }
//                     }
//                 }
//             }
//             compute_step(io, dt, tno);
//             t += dt;
//         }
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
    real hmin = uf(nghost,nghost)[0];
    real hmax = hmin;
    for (int j = nghost; j < ny+nghost; ++j)
        for (int i = nghost; i < nx+nghost; ++i) {
            vec& uij = uf(i,j);
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
