#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <omp.h>

//Forward declaration, because including central2dwrapper.h would be a circular dependency
template <class Physics, class Limiter> class Central2DWrapper;

// An enum type for the copy_corner_ghosts method
enum class Corner { Northeast, Northwest, Southeast, Southwest };

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

    Central2D(real dx, real dy,    // Cell size in x/y
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              real cfl = 0.45) :  // Max allowed CFL number
        nx(nx), ny(ny),
        nx_all(nx + 2*nghost),
        ny_all(ny + 2*nghost),
        dx(dx), dy(dy),
        cfl(cfl), 
        // explanation: 
        // use floats instead of double/real
        // instead of having intermediate "vec" data structure,
        // just use a linear array - improves vectorization
        // as per Prof. Bindel's advice (and C code, although that
        // is different because it allocates a ghost cell buffer
        // while ours doesn't, like the original C++ code)
        u_ ((float *)malloc(sizeof(float) * nx_all * ny_all * 3)),
        f_ ((float *)malloc(sizeof(float) * nx_all * ny_all * 3)),
        g_ ((float *)malloc(sizeof(float) * nx_all * ny_all * 3)),
        ux_((float *)malloc(sizeof(float) * nx_all * ny_all * 3)),
        uy_((float *)malloc(sizeof(float) * nx_all * ny_all * 3)),
        fx_((float *)malloc(sizeof(float) * nx_all * ny_all * 3)),
        gy_((float *)malloc(sizeof(float) * nx_all * ny_all * 3)),
        v_ ((float *)malloc(sizeof(float) * nx_all * ny_all * 3)) {}

    // Advance from time 0 to time tfinal
    void run(const real tfinal);

	// Advance the simulation by one timestep pair from its current time,
	// returning the new value of dt computed over this simulation's grid.
	real take_timestep_pair(const real tfinal);

	
    // Call f(Uxy, x, y) at each cell center to set initial conditions
    template <typename F>
    void init(F f);

	// Initializes cells by copying a block of cells from larger_u
	void init_as_subdomain(const float* larger_u, 
			const int larger_nx_all, const int larger_ny_all, const int x_start, const int y_start);

	// Copy cells to the larger domain that this simulator is a subdomain of
	void copy_results_out(float*  larger_v,
			const int larger_nx_all, const int larger_ny_all, const int x_start, const int y_start);

	// Copy cells to the vertical ghost region of the larger domain
	void copy_vert_ghosts(float*  larger_v,
			const int larger_nx_all, const int larger_ny_all, const int larger_nx,  
			const int y_start, const bool target_left);

	// Copy cells to the horizontal ghost region of the larger domain 		
	void copy_horiz_ghosts(float*  larger_v,
			const int larger_nx_all, const int larger_ny_all, const int larger_ny, 
			const int x_start, const bool target_top);

	// Copy cells from one corner of this domain to the opposite corner ghost region of the larger domain
	void copy_corner_ghosts(float*  larger_v,
		const int larger_nx_all, const int larger_ny_all, const int larger_ny, 
		const int larger_nx, const Corner target);

    // Diagnostics
    void solution_check();

    // Array size accessors
    int xsize() const { return nx; }
    int ysize() const { return ny; }
    
    // Read / write elements of simulation state
    const vec operator()(int i, int j) const {
        vec vr;
        vr[0] = u_[offset(i+nghost,j+nghost,0)];
        vr[1] = u_[offset(i+nghost,j+nghost,1)];
        vr[2] = u_[offset(i+nghost,j+nghost,2)];
        return vr;
    }
    
private:
	//IMPORTANT: This must be equal to 2*Central2DWrapper::nbatch if Central2D is being wrapped,
	//but I can't express that dependency because then the headers depend on each other
    static constexpr int nghost = 4; 	//Number of ghost cells

    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number

    float* u_;            // Solution values
    float* f_;            // Fluxes in x
    float* g_;            // Fluxes in y
    float* ux_;           // x differences of u
    float* uy_;           // y differences of u
    float* fx_;           // x differences of f
    float* gy_;           // y differences of g
    float* v_;            // Solution values at next step

    // changed offset function to reflect our new array layout; analogous to the
    // offset in Professor Bindel's stepper.c (not central2d_offset, see line 228).
    // We used d instead of k to represent dimension (one can think of the state as a 
    // 3-dimensional vector)
    inline int offset(int ix, int iy, short d) const { return ((d*ny_all)+iy)*nx_all+ix; }

    // Array accessor functions
    // We access d first (i.e. 'd' represents the first dimension along the array)
    // because the derivatives of each component are separable and thus
    // computations of each component can be blocked together more easily.
    // This is also in line with the way offset is implemented (the first third
    // of the array is where d = 0, the second is d = 1, and the last is d = 2)
    inline float& u(int ix, int iy, short d)    { return u_[offset(ix,iy,d)]; }
    inline float& v(int ix, int iy, short d)    { return v_[offset(ix,iy,d)]; }
    inline float& f(int ix, int iy, short d)    { return f_[offset(ix,iy,d)]; }
    inline float& g(int ix, int iy, short d)    { return g_[offset(ix,iy,d)]; }
    inline float& ux(int ix, int iy, short d)    { return ux_[offset(ix,iy,d)]; }
    inline float& uy(int ix, int iy, short d)    { return uy_[offset(ix,iy,d)]; }
    inline float& fx(int ix, int iy, short d)    { return fx_[offset(ix,iy,d)]; }
    inline float& gy(int ix, int iy, short d)    { return gy_[offset(ix,iy,d)]; }

    // Wrapped accessor (periodic BC)
    int ioffset(int ix, int iy, short d) {
        return offset( (ix+nx-nghost) % nx + nghost,
                       (iy+ny-nghost) % ny + nghost, d );
    }

    inline float& uwrap(int ix, int iy, short d)  { return u_[ioffset(ix,iy,d)]; }

    // Apply limiter to a single float component
    // Not dealing with vectors anymore - no loop here necessary
    static void limdiff(float& du, const float& um, const float& u0, const float& up) {
        du = Limiter::limdiff(um, u0, up);
    }

    // Stages of the main algorithm
    void apply_periodic();
    void compute_fg_speeds(real& cx, real& cy);
    void limited_derivs();
    void compute_step(int io, real dt);

	// Debugging
	void check_heights(int stepno);
	void print_board();

	//Let Central2DWrapper touch my private members (mostly for debugging)
	friend class Central2DWrapper<Physics, Limiter>;

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
    const int a=nx,b=ny;
    for (int iy = 0; iy < b; ++iy) {
        for (int ix = 0; ix < a; ++ix) {
            vec vr; // Function f requires a vec, not a single float
            vr[0] = u(nghost+ix,nghost+iy,0);
            vr[1] = u(nghost+ix,nghost+iy,1);
            vr[2] = u(nghost+ix,nghost+iy,2);
            f(vr, (ix+0.5)*dx, (iy+0.5)*dy);
            u(nghost+ix,nghost+iy,0) = vr[0];
            u(nghost+ix,nghost+iy,1) = vr[1];
            u(nghost+ix,nghost+iy,2) = vr[2]; // write out result to array
        }
    }
}

/**
 * Initializes cells by copying a block of cells from larger_u, starting at
 * offset start_x, start_y.
 */
template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::init_as_subdomain(const float*  larger_u,
		const int larger_nx_all, const int larger_ny_all, const int x_start, const int y_start)
{
	for (short d = 0; d < 3; d++) {
		for (int y = 0; y < ny_all; ++y) {
			for (int x = 0; x < nx_all; ++x) {
				//Copy starting at x_start and y_start, but include neighboring cells as ghosts
				//Based on offset function, to access larger_u(x,y), I use larger_u[y*larger_nx_all + x]
				u(x,y,d) = larger_u[((d*larger_ny_all) + (y + y_start - nghost)) * larger_nx_all + (x + x_start - nghost)];
			}
		}
	}
}

/**
 * Copy cells to the larger domain that this simulator is a subdomain of
 */
template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::copy_results_out(float*  larger_v,
		const int larger_nx_all, const int larger_ny_all, const int x_start, const int y_start)
{
	for (short d = 0; d < 3; d++) {
		for (int y = 0; y < ny; ++y) {
			for (int x = 0; x < nx; ++x) {
				//Based on offset function, to access larger_v(x,y), I use larger_v[y*larger_nx_all + x]
				larger_v[((d*larger_ny_all) + (y + y_start)) * larger_nx_all + (x + x_start)] = u(x+nghost,y+nghost,d);
			}
		}
	}
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::copy_vert_ghosts(float*  larger_v,
		const int larger_nx_all, const int larger_ny_all, const int larger_nx, 
		const int y_start, const bool target_left)
{
	for (short d = 0; d < 3; d++) {
		for (int y = 0; y < ny; ++y) {
			for (int x = 0; x < nghost; ++x) {
				if(target_left) {
					//For left ghost cells, copy from the right edge of the local (non-ghost) board
					//Conveniently, advancing by nx is the same as advancing by nx+nghost (to the right edge)
					// and then backing up by nghost (to the beginning of the region to be copied)
					larger_v[((d*larger_ny_all) + (y + y_start)) * larger_nx_all + x] = u(x+nx, y+nghost, d);
				} else {
					//For right ghost cells, copy from the left edge of the local (non-ghost) board
					larger_v[((d*larger_ny_all) + (y + y_start)) * larger_nx_all + (larger_nx+nghost+x)] = u(x+nghost, y+nghost, d);
				}
			}
		}
	}
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::copy_horiz_ghosts(float*  larger_v,
		const int larger_nx_all, const int larger_ny_all, const int larger_ny, 
		const int x_start, const bool target_top)
{
	for (short d = 0; d < 3; d++) {
		for (int x = 0; x < nx; ++x) {
			for(int y = 0; y < nghost; ++y) {
				if(target_top) {
					//For top ghost cells, copy from the bottom edge of the local (non-ghost) board
					//Conveniently, advancing by ny is the same as advancing by ny+nghost (to the last row)
					// and then backing up by nghost (to the beginning of the region to be copied)
					larger_v[((d*larger_ny_all) + y) * larger_nx_all + (x+x_start)] = u(x+nghost, y+ny, d);
				} else {
					//For bottom ghost cells, copy from the top edge of the local (non-ghost) board
					larger_v[((d*larger_ny_all) + (larger_ny+nghost+y)) * larger_nx_all + (x+x_start)] = u(x+nghost, y+nghost, d);
				}
			}
		}
	}
}

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::copy_corner_ghosts(float*  larger_v,
		const int larger_nx_all, const int larger_ny_all, const int larger_ny, 
		const int larger_nx, const Corner target)
{
	for (short d = 0; d < 3; d++) {
		for (int x = 0; x < nghost; ++x) {
			for (int y = 0; y < nghost; ++y) {
				switch(target) {
					case Corner::Northwest:
						//Copy from the bottom-right corner of this grid to the top-left corner of the larger grid
						larger_v[((d*larger_ny_all) + y) * larger_nx_all + x] = u(x+nx, y+ny,d);
						break;
					case Corner::Northeast:
						//Copy from bottom-left to top-right 
						larger_v[((d*larger_ny_all) + y) * larger_nx_all + (x+larger_nx+nghost)] = u(x+nghost, y+ny,d);
						break;
					case Corner::Southwest:
						//Copy from top-right to bottom-left
						larger_v[((d*larger_ny_all) + (y+larger_ny+nghost)) * larger_nx_all + x] = u(x+nx, y+nghost,d);
						break;
					case Corner::Southeast:
						//Copy from top-left to bottom-right
						larger_v[((d*larger_ny_all) + (y+larger_ny+nghost)) * larger_nx_all + (x+larger_nx+nghost)] = u(x+nghost, y+nghost,d);
						break;
				}
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
void Central2D<Physics, Limiter>::apply_periodic()
{
    for (short d = 0; d < 3; d++) {
        for (int ix = 0; ix < nghost; ++ix) {
        #pragma ivdep // vectorize this loop (ignore "dependencies")
            for (int iy = 0; iy < ny_all; ++iy) {
                u(ix,          iy,d) = uwrap(ix,          iy,d);
                u(nx+nghost+ix,iy,d) = uwrap(nx+nghost+ix,iy,d);
            }
        }
        for (int iy = 0; iy < nghost; ++iy) {
            #pragma ivdep
            for (int ix = 0; ix < nx_all; ++ix) {
                u(ix,          iy,d) = uwrap(ix,          iy,d);
                u(ix,ny+nghost+iy,d) = uwrap(ix,ny+nghost+iy,d);
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
void Central2D<Physics, Limiter>::compute_fg_speeds(real& cx_, real& cy_)
{
    using namespace std;
    real cx = 1.0e-15;
    real cy = 1.0e-15;
    for (int iy = 0; iy < ny_all; ++iy)
        #pragma ivdep // vectorize this loop
        for (int ix = 0; ix < nx_all; ++ix) {
            real cell_cx, cell_cy;
            Physics::flux(f(ix,iy,0), f(ix,iy,1), f(ix,iy,2),
                          g(ix,iy,0), g(ix,iy,1), g(ix,iy,2),
                          u(ix,iy,0), u(ix,iy,1), u(ix,iy,2));
            Physics::wave_speed(cell_cx, cell_cy, u(ix,iy,0),
                                u(ix,iy,1), u(ix,iy,2));
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
#pragma omp declare simd
void Central2D<Physics, Limiter>::limited_derivs()
{
    for (short d = 0; d < 3; d++) {
        for (int iy = 1; iy < ny_all-1; ++iy ) {
            #pragma ivdep
            //#pragma vector always
            for (int ix = 1; ix < nx_all-1; ++ix) {
                // x derivs
                limdiff( ux(ix,iy,d), u(ix-1,iy,d), u(ix,iy,d), u(ix+1,iy,d) );
                limdiff( fx(ix,iy,d), f(ix-1,iy,d), f(ix,iy,d), f(ix+1,iy,d) );
            }
        }
        for (int iy = 1; iy < ny_all-1; ++iy ) {
            #pragma ivdep
            //#pragma vector always
            for (int ix = 1; ix < nx_all-1; ++ix) {
                limdiff( uy(ix,iy,d), u(ix,iy-1,d), u(ix,iy,d), u(ix,iy+1,d) );
                limdiff( gy(ix,iy,d), g(ix,iy-1,d), g(ix,iy,d), g(ix,iy+1,d) );
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
void Central2D<Physics, Limiter>::compute_step(int batchnum, real dt)
{
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;

    // Predictor (flux values of f and g at half step)
    for (int iy = 1; iy < ny_all-1; ++iy) {
        #pragma ivdep
        for (int ix = 1; ix < nx_all-1; ++ix) {
            float u0 = u(ix,iy,0), u1 = u(ix,iy,1), u2 = u(ix,iy,2);
            u0 -= dtcdx2 * fx(ix,iy,0); // unrolled loops for this
            u0 -= dtcdy2 * gy(ix,iy,0);
            u1 -= dtcdx2 * fx(ix,iy,1);
            u1 -= dtcdy2 * gy(ix,iy,1);
            u2 -= dtcdx2 * fx(ix,iy,2);
            u2 -= dtcdy2 * gy(ix,iy,2);
            Physics::flux(f(ix,iy,0), f(ix,iy,1), f(ix,iy,2),
                          g(ix,iy,0), g(ix,iy,1), g(ix,iy,2),
                          u0, u1, u2);
        }
    }

    // Corrector (finish the step)
	// On an even step, write from nghost-2 to ny+nghost+2 (because it's the first in the batch)
	// On an odd step, write from nghost-1 to ny+nghost-1
	for (short d = 0; d < 3; d++) {
	    for (int iy = nghost-2+batchnum; iy < ny+nghost+2-(3*batchnum); ++iy) {
	        for (int ix = nghost-2+batchnum; ix < nx+nghost+2-(3*batchnum); ++ix) {
                    v(ix,iy,d) =
                        0.2500 * ( u( ix,  iy, d)   + u( ix+1,iy, d) +
                                   u( ix,  iy+1, d) + u( ix+1,iy+1, d)) -
                        0.0625 * ( ux(ix+1,iy,d  ) - ux(ix,  iy, d) +
                                   ux(ix+1,iy+1,d) - ux(ix,  iy+1, d) +
                                   uy(ix,  iy+1, d) - uy(ix,  iy, d) +
                                   uy(ix+1,iy+1,d) - uy(ix+1,iy, d) ) -
                        dtcdx2 * ( f(ix+1,iy, d  ) -  f(ix,  iy, d) +
                                   f(ix+1,iy+1, d) -  f(ix,  iy+1, d)) -
                        dtcdy2 * ( g(ix,  iy+1, d) -  g(ix,  iy, d) +
                                   g(ix+1,iy+1, d) -  g(ix+1,iy, d) );
	        }
	    }
    }

    // Copy from v storage back to main grid
	// On an even step, write from nghost-2 to ny+nghost+2 (because it's the first in the batch)
	// On an odd step, write from nghost to ny+nghost
	const int bo = (batchnum == 0 ? 2 : 0);
	for (short d = 0; d < 3; d++) {
	    for (int j = nghost-bo; j < ny+nghost+bo; ++j){
	        for (int i = nghost-bo; i < nx+nghost+bo; ++i){
	            u(i,j,d) = v(i-batchnum,j-batchnum,d);
	        }
	    }
	}
}

/**
 * Advance the simulation by one timestep pair from its current time,
 * returning the new value of dt computed over this simulation's grid.
 */
template <class Physics, class Limiter>
Central2D<Physics, Limiter>::real Central2D<Physics, Limiter>::take_timestep_pair(const real dt_prev)
{
        real cx, cy;
		real dt_local;
		//Even step
		compute_fg_speeds(cx, cy);
		limited_derivs();
		compute_step(0, dt_prev);
		//Odd step - don't need to do apply_periodic yet if there are enough ghost cells
		compute_fg_speeds(cx, cy);
		limited_derivs();
		compute_step(1, dt_prev);
		//Compute dt at the end instead of at the beginning. Hopefully this doesn't make a difference.
		dt_local = cfl / std::max(cx/dx, cy/dy);
		return dt_local;
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
void Central2D<Physics, Limiter>::run(const real tfinal)
{
	bool done = false;
	real curtime = 0;
	//Since dt is now computed at the end of a batch, 
	//must do an extra step to compute it before the first step
	real cx, cy;
	apply_periodic();
	compute_fg_speeds(cx, cy);
	real dt = cfl / std::max(cx/dx, cy/dy);
	//printf("Starting dt is %f\n", dt);
	while(!done) {
		if (curtime + 2*dt >= tfinal) {
			dt = (tfinal-curtime)/2;
			done = true;
		}
		curtime += 2*dt;
		apply_periodic();
		dt = take_timestep_pair(dt);
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
    real hmin = u(nghost,nghost,0);
    real hmax = hmin;
    for (int j = nghost; j < ny+nghost; ++j)
        for (int i = nghost; i < nx+nghost; ++i) {
            real h = u(i, j, 0); // change these in order to typecheck
            h_sum += h;
            hu_sum += u(i, j, 1);
            hv_sum += u(i, j, 2);
            hmax = max(h, hmax);
            hmin = min(h, hmin);
            assert (h > 0) ;
        }
    real cell_area = dx*dy;
    h_sum *= cell_area;
    hu_sum *= cell_area;
    hv_sum *= cell_area;
    printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n",
           h_sum, hu_sum, hv_sum, hmin, hmax);
}


#endif /* CENTRAL2D_H*/
