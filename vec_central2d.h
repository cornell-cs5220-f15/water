#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cassert>
#include "immintrin.h"

#ifndef NX  
#define NX 200
#endif
 
#ifndef NX_ALL
#define NX_ALL 206
#endif


template <class Physics, class Limiter>
class Central2D {
public:
    typedef typename Physics::real real;
    
    Central2D(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              real cfl = 0.45) :  // Max allowed CFL number
        nx(nx), ny(ny),
        nx_all(nx + 2*nghost),
        ny_all(ny + 2*nghost),
        dx(w/nx), dy(h/ny),
        cfl(cfl) {
	
	}
	/*
	real* u_ = (real*) _mm_malloc(nx_all*ny_all*stride*sizeof(real), 16);
	real* f_ = (real*) _mm_malloc(nx_all*ny_all*stride*sizeof(real), 16);
	real* g_ = (real*) _mm_malloc(nx_all*ny_all*stride*sizeof(real), 16);
	real* ux_ = (real*) _mm_malloc(nx_all*ny_all*stride*sizeof(real), 16);
	real* uy_ = (real*) _mm_malloc(nx_all*ny_all*stride*sizeof(real), 16);
	real* fx_ = (real*) _mm_malloc(nx_all*ny_all*stride*sizeof(real), 16);
	real* gy_ = (real*) _mm_malloc(nx_all*ny_all*stride*sizeof(real), 16);
	real* v_ = (real*) _mm_malloc(nx_all*ny_all*stride*sizeof(real), 16);

	}

    ~Central2D()  {
	_mm_free(u_); _mm_free(f_); _mm_free(g_); _mm_free(ux_);
	_mm_free(uy_); _mm_free(fx_); _mm_free(gy_); _mm_free(v_);
	}
	*/

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
    real*       operator()(int i, int j) {
        return u_ + offset(i+nghost,j+nghost);
    }
    
    const real* operator()(int i, int j) const {
        return u_ + offset(i+nghost,j+nghost);
    }
    
private:
    static constexpr int nghost = 3;   // Number of ghost cells

    static const int stride = 3;		// Accounting for number of elements/cell
    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number

    real u_[NX_ALL*NX_ALL*stride] __attribute__((aligned(32)));            // Solution values
    real f_[NX_ALL*NX_ALL*stride] __attribute__((aligned(32)));            // Fluxes in x
    real g_[NX_ALL*NX_ALL*stride] __attribute__((aligned(32)));            // Fluxes in y
    real ux_[NX_ALL*NX_ALL*stride] __attribute__((aligned(32)));           // x differences of u
    real uy_[NX_ALL*NX_ALL*stride] __attribute__((aligned(32)));           // y differences of u
    real fx_[NX_ALL*NX_ALL*stride] __attribute__((aligned(32)));           // x differences of f
    real gy_[NX_ALL*NX_ALL*stride] __attribute__((aligned(32)));           // y differences of g
    real v_[NX_ALL*NX_ALL*stride] __attribute__((aligned(32)));            // Solution values at next step

//    real  *u_, *f_, *g_, *ux_, *uy_, *fx_, *gy_, *v_;

    // Array accessor functions

    int offset(int ix, int iy) const { return stride*iy*nx_all + stride*ix; }

    real* u(int ix, int iy)    { return u_ + offset(ix,iy); }
    real* v(int ix, int iy)    { return v_ + offset(ix,iy); }
    real* f(int ix, int iy)    { return f_ + offset(ix,iy); }
    real* g(int ix, int iy)    { return g_ + offset(ix,iy); }

    real* ux(int ix, int iy)   { return ux_ + offset(ix,iy); }
    real* uy(int ix, int iy)   { return uy_ + offset(ix,iy); }
    real* fx(int ix, int iy)   { return fx_ + offset(ix,iy); }
    real* gy(int ix, int iy)   { return gy_ + offset(ix,iy); }

    // Wrapped accessor (periodic BC)
    int ioffset(int ix, int iy) {
        return offset( (ix+nx-nghost) % nx + nghost,
                       (iy+ny-nghost) % ny + nghost );
    }

    real* uwrap(int ix, int iy)  { return u_ + ioffset(ix,iy); }

    // Apply limiter to all components in a vector
    static void limdiff(real* du, const real* um, const real* u0, const real* up) {
        for (int m = 0; m < stride; ++m)
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
		for (int m = 0; m < stride; ++m)  {
	            u(ix,          iy)[m] = uwrap(ix,          iy)[m];
       	 	    u(nx+nghost+ix,iy)[m] = uwrap(nx+nghost+ix,iy)[m];
	        }
	}
    // Copy data between top and bottom boundaries
    for (int ix = 0; ix < nx_all; ++ix)
        for (int iy = 0; iy < nghost; ++iy) {
		for (int m = 0; m < stride; ++m)  {
	            u(ix,          iy)[m] = uwrap(ix,          iy)[m];
	      	    u(ix,ny+nghost+iy)[m] = uwrap(ix,ny+nghost+iy)[m];
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
void Central2D<Physics, Limiter>::compute_fg_speeds(real& cx_, real& cy_)
{
    using namespace std;
    real cx = 1.0e-15;
    real cy = 1.0e-15;
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            real cell_cx, cell_cy;
            Physics::flux(f(ix, iy), g(ix, iy), u(ix, iy));
            Physics::wave_speed(cell_cx, cell_cy, u(ix, iy));
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



template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::compute_step(int io, real dt)
{
	real dtcdx2 = 0.5 * dt / dx;
	__m256 dtdxr = _mm256_broadcast_ss(&dtcdx2);
	real dtcdy2 = 0.5 * dt / dy;
	__m256 dtdyr = _mm256_broadcast_ss(&dtcdy2);

	real utemp[8] __attribute__((aligned(32)));

	// Predictor (flux values of f and g at half step)
	for (int iy = 1; iy < ny_all-1; ++iy)
	for (int ix = 1; ix < nx_all-1; ix+=2) {

	//load 8 floating point values in the column
	 __m256 ur = _mm256_load_ps(u(ix,iy));
	 __m256 fxr = _mm256_load_ps(fx(ix,iy));
	 __m256 gyr = _mm256_load_ps(gy(ix,iy));

	fxr = _mm256_mul_ps(fxr, dtdxr);
	gyr = _mm256_mul_ps(gyr, dtdyr);

	ur = _mm256_sub_ps(ur, fxr);
	ur = _mm256_sub_ps(ur, gyr);

	_mm256_store_ps(utemp, ur);

	Physics::flux(f(ix,iy), g(ix,iy), utemp);
	Physics::flux(f(ix+1,iy), g(ix+1,iy), utemp+3);
	}

    // Corrector (finish the step)
    for (int iy = nghost-io; iy < ny+nghost-io; ++iy)
        for (int ix = nghost-io; ix < nx+nghost-io; ++ix) {
            for (int m = 0; m < stride; ++m) {
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
		for (int m = 0; m < stride; ++m)  {
	            u(i,j)[m] = v(i-io,j-io)[m];
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
            real* uij = u(i,j);
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
