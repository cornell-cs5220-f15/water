#ifndef CENTRAL2DVEC_H
#define CENTRAL2DVEC_H

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
    typedef typename Physics::iter iter;
    
    Central2D(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              int nfield,    // Max allowed CFL number
              real cfl = 0.45):
    nx(nx), ny(ny),
	nx_all(2*ng+nx), ny_all(2*ng+ny),
	nfield(nfield),
    dx(w/nx), dy(h/ny),
	cfl(cfl),
	N(nfield * (nx + 2*ng)* (ny + 2*ng)),
	all(8*N), pointers(8) {}

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
  /*  vec&  operator()(int i, int j) {
        return u_[offset(i+nghost,j+nghost)];
    }
    
    const vec& operator()(int i, int j) const {
        return u_[offset(i+nghost,j+nghost)];
    }*/
    
private:
    static constexpr int ng = 3;   // Number of ghost cells
    const int nx_all,ny_all, nfield;
    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number
    const int N;
    
    std::vector<real> all;            // Solution values
 
    std::vector<iter> pointers;


    
    
    
    // Array accessor functions

    int offset(int k, int ix, int iy) const { return (k*ny_all+(ng+iy))*nx_all+(ng+ix); }

  /*  iter& u(int k ,int ix, int iy)    { return *(pointers[0]+offset(k,ix,iy)); }
    vec& v(int ix, int iy)    { return v_[offset(ix,iy)]; }
    vec& f(int ix, int iy)    { return f_[offset(ix,iy)]; }
    vec& g(int ix, int iy)    { return g_[offset(ix,iy)]; }

    vec& ux(int ix, int iy)   { return ux_[offset(ix,iy)]; }
    vec& uy(int ix, int iy)   { return uy_[offset(ix,iy)]; }
    vec& fx(int ix, int iy)   { return fx_[offset(ix,iy)]; }
    vec& gy(int ix, int iy)   { return gy_[offset(ix,iy)]; }*/

    // Wrapped accessor (periodic BC)
    int ioffset(int ix, int iy) {
        return offset( (ix+nx-ng) % nx + ng,
                       (iy+ny-ng) % ny + ng );
    }

   // vec& uwrap(int ix, int iy)  { return u_[ioffset(ix,iy)]; }

    // Apply limiter to all components in a vector
 /*   static void limdiff(vec& du, const vec& um, const vec& u0, const vec& up) {
        for (int m = 0; m < du.size(); ++m)
            du[m] = Limiter::limdiff(um[m], u0[m], up[m]);
    }*/

    // Stages of the main algorithm
    void apply_periodic(iter u,
                        int nx, int ny, int ng, int nfield);
    void compute_fg_speeds(real& cx, real& cy);
    void limited_derivs();
    void compute_step(int io, real dt);
    void copy_subgrid(iter dst, iter src,
                      int nx, int ny, int stride);
    void limited_deriv1(iter du,const iter u,int ncell);
    void limited_derivk(iter du,const iter u,int ncell, int stride);
    void limited_derivs(iter ux, iter uy,
                        iter fx, iter gy,
                        const iter u,
                        const iter f,
                        const iter g,
                        int nx, int ny, int nfield);
    void central2d_predict(iter v,
                           const iter u,
                           const iter fx,
                           const iter gy,
                           float dtcdx2, float dtcdy2,
                           int nx, int ny, int nfield);
    void central2d_correct(iter v,
                           const iter u,
                           const iter ux,
                           const iter uy,
                           const iter f,
                           const iter g,
                           float dtcdx2, float dtcdy2,
                           int xlo, int xhi, int ylo, int yhi,
                           int nx, int ny, int nfield);
    void compute_step(iter u, iter v,
                      iter ux, iter uy,
                      iter f, iter fx,
                      iter g, iter gy,
                      int io, int nx, int ny, int ng,
                      int nfield,
                      real dt, real dx, real dy);
    int xrun(iter u, iter v,
         iter ux, iter uy,
         iter f, iter fx,
         iter g, iter gy,
         int nx, int ny, int ng,
         int nfield, float tfinal, real dx, real dy,
         real cfl);
    int compute_step(float tfinal);


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
//////CHANGE!!!!!!
template <class Physics, class Limiter>
template <typename F>
void Central2D<Physics, Limiter>::init(F f)
{
    pointers[0] = all.begin();
    pointers[1] = all.begin() + N;
    pointers[2] = all.begin() + 2*N;
    pointers[3] = all.begin() + 3*N;
    pointers[4] = all.begin() + 4*N;
    pointers[5] = all.begin() + 5*N;
    pointers[6] = all.begin() + 6*N;
    pointers[7] = all.begin() + 7*N;
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix){
            int ind=(ng+iy)*nx_all+(ng+ix);
            f(pointers[0]+ind, (ix+0.5)*dx, (iy+0.5)*dy, nx_all*ny_all);
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
void Central2D<Physics, Limiter>::copy_subgrid(iter dst,
                  iter src,
                  int nx, int ny, int stride)
{
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix)
            *(dst+(iy*stride+ix)) = *(src + (iy*stride+ix));
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::apply_periodic(iter u,
                                                 int nx, int ny, int ng, int nfield)
{
    int s = nx + 2*ng;
    int field_stride = (ny+2*ng)*s;
    // Offsets of left, right, top, and bottom data blocks and ghost blocks
    int l = nx,   lg = 0;
    int r = ng,   rg = nx+ng;
    int b = ny*s, bg = 0;
    int t = ng*s, tg = (nx+ng)*s;
    
    // Copy data into ghost cells on each side
    for (int k = 0; k < nfield; ++k) {
        iter uk = u + k*field_stride;
        copy_subgrid(uk+lg, uk+l, ng, ny+2*ng, s);
        copy_subgrid(uk+rg, uk+r, ng, ny+2*ng, s);
        copy_subgrid(uk+tg, uk+t, nx+2*ng, ng, s);
        copy_subgrid(uk+bg, uk+b, nx+2*ng, ng, s);
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

/*template <class Physics, class Limiter>
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
}*/

/**
 * ### Derivatives with limiters
 * 
 * In order to advance the time step, we also need to estimate
 * derivatives of the fluxes and the solution values at each cell.
 * In order to maintain stability, we apply a limiter here.
 */

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::limited_deriv1(iter du,const iter u,int ncell)
{
    for (int i=0; i<ncell; ++i)
        (*(du+i))=limdiff(*(u+i-1),*(u+i),*(u+i+1));
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::limited_derivk(iter du,const iter u,int ncell, int stride)
{
    assert(stride>0);
    for (int i=0; i<ncell; ++i)
        (*(du+i))=limdiff(*(u+i-stride),*(u+i),*(u+i+stride));
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::limited_derivs(iter ux, iter uy,
                                                 iter fx, iter gy,
                                                 const iter u,
                                                 const iter f,
                                                 const iter g,
                                                 int nx, int ny, int nfield)
{
    for (int k = 0; k < nfield; ++k)
        for (int iy = 1; iy < ny-1; ++iy) {
            int offset = (k*ny+iy)*nx+1;
            limited_deriv1(ux+offset, u+offset, nx-2);
            limited_deriv1(fx+offset, f+offset, nx-2);
            limited_derivk(uy+offset, u+offset, nx-2, nx);
            limited_derivk(gy+offset, g+offset, nx-2, nx);
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
void Central2D<Physics, Limiter>::central2d_predict(iter v,
                       const iter u,
                       const iter fx,
                       const iter gy,
                       float dtcdx2, float dtcdy2,
                       int nx, int ny, int nfield)
{
    for (int k = 0; k < nfield; ++k)
        for (int iy = 1; iy < ny-1; ++iy)
            for (int ix = 1; ix < nx-1; ++ix) {
                int offset = (k*ny+iy)*nx+ix;
                *(v+offset) = *(u+offset) -
                dtcdx2 * (*(fx+offset)) -
                dtcdy2 * (*(gy+offset));
            }
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::central2d_correct(iter v,
                       const iter u,
                       const iter ux,
                       const iter uy,
                       const iter f,
                       const iter g,
                       float dtcdx2, float dtcdy2,
                       int xlo, int xhi, int ylo, int yhi,
                       int nx, int ny, int nfield)
{
    assert(0 <= xlo && xlo < xhi && xhi <= nx);
    assert(0 <= ylo && ylo < yhi && yhi <= ny);
    
    for (int k = 0; k < nfield; ++k)
        for (int iy = ylo; iy < yhi; ++iy)
            for (int ix = xlo; ix < xhi; ++ix) {
                
                int j00 = (k*ny+iy)*nx+ix;
                int j10 = j00+1;
                int j01 = j00+nx;
                int j11 = j00+nx+1;
                
                *(v+j00) =
                0.2500f * ( (*(u+j00)) + (*(u+j01)) + (*(u+j10)) + (*(u+j11)) ) -
                0.0625f * ( (*(ux+j10)) - (*(ux+j00)) +
                           (*(ux+j11)) - (*(ux+j01)) +
                           (*(uy+j01)) - (*(uy+j00)) +
                           (*(uy[j11])) - (*(uy+j10)) ) -
                dtcdx2  * ( (*(f+j10)) - (*(f+j00)) +
                           (*(f+j11)) - (*(f+j01)) ) -
                dtcdy2  * ( (*(g+j01)) - (*(g+j00)) +
                           (*(g+j11)) - (*(g+j10)) );
            }
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::compute_step(iter u, iter v,
                                               iter ux, iter uy,
                                               iter f, iter fx,
                                               iter g, iter gy,
                                               int io, int nx, int ny, int ng,
                                               int nfield,
                                               real dt, real dx, real dy)
{
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    
    float dtcdx2 = 0.5 * dt / dx;
    float dtcdy2 = 0.5 * dt / dy;
    
    Physics::flux(f, g, u, nx_all * ny_all, nx_all * ny_all);
    limited_derivs(ux,uy,fx,gy,u,f,g,nx_all,ny_all,nfield);
    central2d_predict(v, u, fx, gy, dtcdx2, dtcdy2,
                      nx_all, ny_all, nfield);
    
    // Flux values of f and g at half step
    for (int iy = 1; iy < ny_all-1; ++iy) {
        int jj = iy*nx_all+1;
        Physics::flux(f+jj, g+jj, v+jj, nx_all-2, nx_all * ny_all);
    }
    
    central2d_correct(v, u, ux, uy, f, g, dtcdx2, dtcdy2,
                      ng-io, nx+ng-io,
                      ng-io, ny+ng-io,
                      nx_all, ny_all, nfield);
    
    // Copy from v storage back to main grid
    for (int k = 0; k < nfield; ++k)
        std::copy(v+(k*ny_all+ng-io)*nx_all+ng-io,
                  v+(k*ny_all+ng-io)*nx_all+ng-io+(ny * nx_all),u+(k*ny_all+ng   )*nx_all+ng);

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
int Central2D<Physics, Limiter>::xrun(iter u, iter v,
                                       iter ux, iter uy,
                                       iter f, iter fx,
                                       iter g, iter gy,
                                       int nx, int ny, int ng,
                                       int nfield, float tfinal, real dx, real dy,
                                       real cfl)
{
    int nstep=0;
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    bool done = false;

    real t = 0;
    while (!done) {
        real dt;
        for (int io = 0; io < 2; ++io) {
            real cx, cy;
            apply_periodic(u,nx,ny,ng,nfield);
            Physics::wave_speed(cx,cy,u,nx_all * ny_all, nx_all * ny_all);
            limited_derivs();
            if (io == 0) {
                dt = cfl / fmax(cx/dx, cy/dy);
                if (t + 2*dt >= tfinal) {
                    dt = (tfinal-t)/2;
                    done = true;
                }
            }
  
            compute_step(u,v, ux,uy,f,fx,g,gy,io,nx,ny,ng,nfield,dt,dx,dy);
            t += dt;
            nstep +=2;
        }
    }
    return nstep;
}

template <class Physics, class Limiter>
int Central2D<Physics, Limiter>::compute_step(float tfinal)
{
    return central2d_xrun(pointers[0], pointers[1], pointers[2], pointers[3],
                          pointers[4], pointers[5], pointers[6], pointers[7],
                          nx, ny, ng,
                          nfield,tfinal, dx, dy, cfl);
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

/*template <class Physics, class Limiter>
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
}*/

//ldoc off
#endif /* CENTRAL2D_H*/
