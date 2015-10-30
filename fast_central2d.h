#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#include <cstdio>
#include <cmath>
#include <cassert>
#include <array>
#include <omp.h>

#ifndef NX  
#define NX 400
#endif

#ifndef BOUND
#define BOUND 16
#endif

//ldoc on

template <class Physics, class Limiter>
class Central2D {
public:
    typedef float real;

    Central2D(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              real cfl = 0.45) :  // Max allowed CFL number
			dx(w/nx), dy(h/ny),
			cfl(cfl)
		{}


    // Advance from time 0 to time tfinal
    void run(real tfinal);

    // Call f(Uxy, x, y) at each cell center to set initial conditions
    void init();

    // Diagnostics
    void solution_check();

    // Array size accessors
    int xsize() const { return nx; }
    int ysize() const { return ny; }
    
    // Read / write elements of simulation state
    real&       operator()(int i, int j) {
        return u1_[i + nx*j];
    }
    
    const real& operator()(int i, int j) const {
        return u1_[i + nx*j];
    }
 
    // Array accessor functions for complete board
    inline int offset(int ix, int iy)	const { return iy*nx+ix; }
    inline real& u1(int ix, int iy)		{ return u1_[offset(ix,iy)]; }
    inline real& u2(int ix, int iy)		{ return u2_[offset(ix,iy)]; }
    inline real& u3(int ix, int iy)		{ return u3_[offset(ix,iy)]; }

	// Array accessor functions for individual domains
	inline int rel_offset(int ix, int iy)	{ return iy*ndx_all+ix; }
  
private:
    // Gravitational force (compile time constant)
   static constexpr real g = 9.8;

   static constexpr int nghost = 3;   // Number of ghost cells
	static constexpr int nsteps = 1;	// Successive number of timesteps evaluated without sync
	static constexpr int ndomainx = 4;
	static constexpr int ndomainy = 4;

	// Number of (non-ghost) cells in x/y
	const int nx = NX;
	const int ny = NX;

	const int ndx = ceil(NX/ndomainx);
	const int ndy = ceil(NX/ndomainy);
	const int ndx_all = ndx + 2*nghost;
	const int ndy_all = ndy + 2*nghost;

	const real dx, dy;         // Cell size in x/y
	const real cfl;            // Allowed CFL number

	// Main grid Solution values
  	real u1_[NX*NX] __attribute__((aligned(BOUND))); 
  	real u2_[NX*NX] __attribute__((aligned(BOUND))); 
  	real u3_[NX*NX] __attribute__((aligned(BOUND))); 

    /* Stages of the main algorithm
	void init_domain(int t_id);
    void copy_halo(int t_id);
    void compute_speeds(real& cx, real& cy);
	void compute_flux();
    void limited_derivs();
	void collect_domains(int t_id);
    void compute_step(int io, real dt);
	*/
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
void Central2D<Physics, Limiter>::init()
{
	//default is dam break initial condition to generate the final image. 
    //#pragma omp parallel for 
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix){
            real x = (ix+0.5)*dx;
            real y = (iy+0.5)*dy;
            x -= 1;
            y -= 1;
            u1(ix,iy) = 1.0 + 0.5*(x*x + y*y < 0.25+1e-5);
            u2(ix,iy) = 0;
            u3(ix,iy) = 0;
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
	//explicitly turn off dynamic threads
	omp_set_dynamic(0);
	omp_set_num_threads(ndomainx*ndomainy);
	
	real t, dt, dtcdx2, dtcdy2;
	real cx = 1.0e-15;
	real cy = 1.0e-15;
	bool done;
	#pragma omp parallel shared(dt,t,done,dtcdx2,dtcdy2,cx,cy)
	{
		// Domain solution values
		real u1_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real u2_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real u3_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;

		// Fluxes in x
		real f1_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real f2_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real f3_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;

		// Fluxes in y
		real g1_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real g2_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real g3_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;

		// x differences of u
		real ux1_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real ux2_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real ux3_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;

		// y differences of u
		real uy1_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real uy2_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real uy3_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;

		// x differences of f
		real fx1_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real fx2_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real fx3_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;

		// y differences of g
		real gy1_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real gy2_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ; 
		real gy3_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ; 

		// Solution values at next step
		real v1_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real v2_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
		real v3_d[ndx_all*ndy_all] __attribute__((aligned(BOUND))) ;
	
		int t_id = omp_get_thread_num();

		// Initialize individual domains
		int row_offset = ceil(t_id % ndomainx);
		row_offset *= ndx;
		int col_offset = ceil(t_id / ndomainx);
		col_offset *= ndy;


		// Divide data for each thread from the  main grid
		for (int iy = 0; iy < ndy; ++iy)
			for (int ix = 0; ix < ndx; ++ix) {
					u1_d[rel_offset(nghost+ix,nghost+iy)] = u1(row_offset+ix, col_offset+iy);
					u2_d[rel_offset(nghost+ix,nghost+iy)] = u2(row_offset+ix, col_offset+iy);
					u3_d[rel_offset(nghost+ix,nghost+iy)] = u3(row_offset+ix, col_offset+iy);
			}
	
		#pragma omp barrier
		#pragma omp flush (done, t)
		#pragma omp critical
		{
			done = false;
			t = 0;
		}

		while (!done) {

			//compute_speeds(cx, cy);
			real priv_cx = 1.0e-15;
			real priv_cy = 1.0e-15;
			for (int iy = nghost; iy < ndy+nghost; ++iy)
				for (int ix = nghost; ix < ndx+nghost; ++ix) {
					real cell_cx, cell_cy;
					real h = u1_d[rel_offset(ix,iy)], hu = u2_d[rel_offset(ix,iy)], hv = u3_d[rel_offset(ix,iy)];
					real root_gh = sqrt(g * h);  // NB: Don't let h go negative!

					cell_cx = abs(hu/h) + root_gh;
					cell_cy = abs(hv/h) + root_gh;

					priv_cx = std::max(priv_cx, cell_cx);
					priv_cy = std::max(priv_cy, cell_cy);
				}

			//reduction
			#pragma omp flush(cx,cy)
			if (priv_cx > cx)	{
				#pragma omp critical
				{
					if (priv_cx > cx) cx = priv_cx;
				}
			}
			if (priv_cy > cy)	{
				#pragma omp critical
				{
					if (priv_cy > cy) cy = priv_cy;
				}
			}

			//if (t_id == 0)
			#pragma omp flush(cx,cy,t,dt,dtcdx2,dtcdy2,done)
			#pragma omp critical
			{
				dt = cfl / std::max(cx/dx, cy/dy);
				dtcdx2 = 0.5 * dt / dx;
				dtcdy2 = 0.5 * dt / dy;

				if (t + 2*(dt)*nsteps >= tfinal) {
					dt = (tfinal-t)/2.0f;
					done = true;
				}
			}

				//copy_halo(t_id);			
				// Copy data between top and bottom boundaries
				for (int iy = 0; iy < ndy_all; ++iy)
					for (int ix = 0; ix < nghost; ++ix) {
				
						u1_d[rel_offset(ix,           iy)] = u1((row_offset+ix+nx-nghost)%nx, (col_offset+iy+ny-nghost)%ny);
						u2_d[rel_offset(ix,           iy)] = u2((row_offset+ix+nx-nghost)%nx, (col_offset+iy+ny-nghost)%ny);
						u3_d[rel_offset(ix,           iy)] = u3((row_offset+ix+nx-nghost)%nx, (col_offset+iy+ny-nghost)%ny);

						u1_d[rel_offset(ndx+nghost+ix,iy)] = u1((row_offset+ix+nx+ndx)%nx, (col_offset+iy+ny-nghost)%ny);
						u2_d[rel_offset(ndx+nghost+ix,iy)] = u2((row_offset+ix+nx+ndx)%nx, (col_offset+iy+ny-nghost)%ny);
						u3_d[rel_offset(ndx+nghost+ix,iy)] = u3((row_offset+ix+nx+ndx)%nx, (col_offset+iy+ny-nghost)%ny);
					}
				// Copy data between right and left boundaries
				for (int ix = 0; ix < ndx_all; ++ix)
					for (int iy = 0; iy < nghost; ++iy) {
						u1_d[rel_offset(ix,           iy)] = u1((row_offset+ix+nx-nghost)%nx, (col_offset+iy+ny-nghost)%ny);
						u2_d[rel_offset(ix,           iy)] = u2((row_offset+ix+nx-nghost)%nx, (col_offset+iy+ny-nghost)%ny);
						u3_d[rel_offset(ix,           iy)] = u3((row_offset+ix+nx-nghost)%nx, (col_offset+iy+ny-nghost)%ny);

						u1_d[rel_offset(ix,ndy+nghost+iy)] = u1((row_offset+ix+nx-nghost)%nx, (col_offset+iy+ny+ndy)%ny);
						u2_d[rel_offset(ix,ndy+nghost+iy)] = u2((row_offset+ix+nx-nghost)%nx, (col_offset+iy+ny+ndy)%ny);
						u3_d[rel_offset(ix,ndy+nghost+iy)] = u3((row_offset+ix+nx-nghost)%nx, (col_offset+iy+ny+ndy)%ny);

					}

			#pragma omp barrier

			for (int step = 0; step < 2*nsteps; ++step) {
				int io = step % 2;

				//compute_flux();
				for (int iy = 0; iy < ndy_all; ++iy)
					for (int ix = 0; ix < ndx_all; ++ix) {
						real h = u1_d[rel_offset(ix,iy)], hu = u2_d[rel_offset(ix,iy)], hv = u3_d[rel_offset(ix,iy)];

						f1_d[rel_offset(ix,iy)] = hu;
						f2_d[rel_offset(ix,iy)] = hu*hu/h + (0.5*g)*h*h;
						f3_d[rel_offset(ix,iy)] = hu*hv/h;

						g1_d[rel_offset(ix,iy)] = hv;
						g2_d[rel_offset(ix,iy)] = hu*hv/h;
						g3_d[rel_offset(ix,iy)] = hv*hv/h + (0.5*g)*h*h;

					}
	

				//limited_derivs():
				//#pragma simd
				for (int iy = 1; iy < ndy_all-1; ++iy)
					for (int ix = 1; ix < ndx_all-1; ++ix) {
					   // Apply limiter to all components in a vector
						// x derivs
						ux1_d[rel_offset(ix,iy)] = Limiter::limdiff(u1_d[rel_offset(ix-1,iy)], u1_d[rel_offset(ix,iy)], u1_d[rel_offset(ix+1,iy)]);
						fx1_d[rel_offset(ix,iy)] = Limiter::limdiff(f1_d[rel_offset(ix-1,iy)], f1_d[rel_offset(ix,iy)], f1_d[rel_offset(ix+1,iy)]);
						// y derivs
						uy1_d[rel_offset(ix,iy)] = Limiter::limdiff(u1_d[rel_offset(ix,iy-1)], u1_d[rel_offset(ix,iy)], u1_d[rel_offset(ix,iy+1)]);
						gy1_d[rel_offset(ix,iy)] = Limiter::limdiff(g1_d[rel_offset(ix,iy-1)], g1_d[rel_offset(ix,iy)], g1_d[rel_offset(ix,iy+1)]);

						// x derivs
						ux2_d[rel_offset(ix,iy)] = Limiter::limdiff(u2_d[rel_offset(ix-1,iy)], u2_d[rel_offset(ix,iy)], u2_d[rel_offset(ix+1,iy)]);
						fx2_d[rel_offset(ix,iy)] = Limiter::limdiff(f2_d[rel_offset(ix-1,iy)], f2_d[rel_offset(ix,iy)], f2_d[rel_offset(ix+1,iy)]);
						// y derivs
						uy2_d[rel_offset(ix,iy)] = Limiter::limdiff(u2_d[rel_offset(ix,iy-1)], u2_d[rel_offset(ix,iy)], u2_d[rel_offset(ix,iy+1)]);
						gy2_d[rel_offset(ix,iy)] = Limiter::limdiff(g2_d[rel_offset(ix,iy-1)], g2_d[rel_offset(ix,iy)], g2_d[rel_offset(ix,iy+1)]);

						// x derivs							
						ux3_d[rel_offset(ix,iy)] = Limiter::limdiff(u3_d[rel_offset(ix-1,iy)], u3_d[rel_offset(ix,iy)], u3_d[rel_offset(ix+1,iy)]);
						fx3_d[rel_offset(ix,iy)] = Limiter::limdiff(f3_d[rel_offset(ix-1,iy)], f3_d[rel_offset(ix,iy)], f3_d[rel_offset(ix+1,iy)]);
						// y derivs
						uy3_d[rel_offset(ix,iy)] = Limiter::limdiff(u3_d[rel_offset(ix,iy-1)], u3_d[rel_offset(ix,iy)], u3_d[rel_offset(ix,iy+1)]);
						gy3_d[rel_offset(ix,iy)] = Limiter::limdiff(g3_d[rel_offset(ix,iy-1)], g3_d[rel_offset(ix,iy)], g3_d[rel_offset(ix,iy+1)]);
					}


				//compute_step(io, dt);	
				// Predictor (flux values of f and g at half step)
				//#pragma simd
				for (int iy = 1; iy < ndy_all-1; ++iy)
					for (int ix = 1; ix < ndx_all-1; ++ix) {
						real h = u1_d[rel_offset(ix,iy)], hu = u2_d[rel_offset(ix,iy)], hv = u3_d[rel_offset(ix,iy)];

						h = h - dtcdx2 * fx1_d[rel_offset(ix,iy)] - dtcdy2 * gy1_d[rel_offset(ix,iy)];
						hu = hu - dtcdx2 * fx2_d[rel_offset(ix,iy)] - dtcdy2 * gy2_d[rel_offset(ix,iy)];
						hv = hv - dtcdx2 * fx3_d[rel_offset(ix,iy)] - dtcdy2 * gy3_d[rel_offset(ix,iy)];

						f1_d[rel_offset(ix,iy)] = hu;
						f2_d[rel_offset(ix,iy)] = hu*hu/h + (0.5*g)*h*h;
						f3_d[rel_offset(ix,iy)] = hu*hv/h;

						g1_d[rel_offset(ix,iy)] = hv;
						g2_d[rel_offset(ix,iy)] = hu*hv/h;
						g3_d[rel_offset(ix,iy)] = hv*hv/h + (0.5*g)*h*h;

				}

				// Corrector (finish the step)
				//#pragma simd
				for (int iy = 1-io; iy < ndy_all-1-io; ++iy)
					for (int ix = 1-io; ix < ndx_all-1-io; ++ix) {
						v1_d[rel_offset(ix,iy)] =
								0.2500 * ( 	u1_d[rel_offset(ix,  iy)] + u1_d[rel_offset(ix+1,iy  )] +
												u1_d[rel_offset(ix,iy+1)] + u1_d[rel_offset(ix+1,iy+1)] ) -
								0.0625 * ( 	ux1_d[rel_offset(ix+1,iy  )] - ux1_d[rel_offset(ix,iy  )] +
												ux1_d[rel_offset(ix+1,iy+1)] - ux1_d[rel_offset(ix,iy+1)] +
												uy1_d[rel_offset(ix,  iy+1)] - uy1_d[rel_offset(ix,  iy)] +
												uy1_d[rel_offset(ix+1,iy+1)] - uy1_d[rel_offset(ix+1,iy)] ) -
								dtcdx2 * ( 	f1_d[rel_offset(ix+1,iy  )] - f1_d[rel_offset(ix,iy  )] +
												f1_d[rel_offset(ix+1,iy+1)] - f1_d[rel_offset(ix,iy+1)] ) -
								dtcdy2 * ( 	g1_d[rel_offset(ix,  iy+1)] - g1_d[rel_offset(ix,  iy)] +
												g1_d[rel_offset(ix+1,iy+1)] - g1_d[rel_offset(ix+1,iy)] );

						v2_d[rel_offset(ix,iy)] =
								0.2500 * ( 	u2_d[rel_offset(ix,  iy)] + u2_d[rel_offset(ix+1,iy  )] +
												u2_d[rel_offset(ix,iy+1)] + u2_d[rel_offset(ix+1,iy+1)] ) -
								0.0625 * ( 	ux2_d[rel_offset(ix+1,iy  )] - ux2_d[rel_offset(ix,iy  )] +
												ux2_d[rel_offset(ix+1,iy+1)] - ux2_d[rel_offset(ix,iy+1)] +
												uy2_d[rel_offset(ix,  iy+1)] - uy2_d[rel_offset(ix,  iy)] +
												uy2_d[rel_offset(ix+1,iy+1)] - uy2_d[rel_offset(ix+1,iy)] ) -
								dtcdx2 * ( 	f2_d[rel_offset(ix+1,iy  )] - f2_d[rel_offset(ix,iy  )] +
												f2_d[rel_offset(ix+1,iy+1)] - f2_d[rel_offset(ix,iy+1)] ) -
								dtcdy2 * ( 	g2_d[rel_offset(ix,  iy+1)] - g2_d[rel_offset(ix,  iy)] +
												g2_d[rel_offset(ix+1,iy+1)] - g2_d[rel_offset(ix+1,iy)] );

						v3_d[rel_offset(ix,iy)] =
								0.2500 * ( 	u3_d[rel_offset(ix,  iy)] + u3_d[rel_offset(ix+1,iy  )] +
												u3_d[rel_offset(ix,iy+1)] + u3_d[rel_offset(ix+1,iy+1)] ) -
								0.0625 * ( 	ux3_d[rel_offset(ix+1,iy  )] - ux3_d[rel_offset(ix,iy  )] +
												ux3_d[rel_offset(ix+1,iy+1)] - ux3_d[rel_offset(ix,iy+1)] +
												uy3_d[rel_offset(ix,  iy+1)] - uy3_d[rel_offset(ix,  iy)] +
												uy3_d[rel_offset(ix+1,iy+1)] - uy3_d[rel_offset(ix+1,iy)] ) -
								dtcdx2 * ( 	f3_d[rel_offset(ix+1,iy  )] - f3_d[rel_offset(ix,iy  )] +
												f3_d[rel_offset(ix+1,iy+1)] - f3_d[rel_offset(ix,iy+1)] ) -
								dtcdy2 * ( 	g3_d[rel_offset(ix,  iy+1)] - g3_d[rel_offset(ix,  iy)] +
												g3_d[rel_offset(ix+1,iy+1)] - g3_d[rel_offset(ix+1,iy)] );

					}

					// Copy from v storage back to the domain grid
					for (int j = 1; j < ndy_all; ++j)
						for (int i = 1; i < ndx_all; ++i)	{
							u1_d[rel_offset(i,j)] = v1_d[rel_offset(i-io,j-io)];
							u2_d[rel_offset(i,j)] = v2_d[rel_offset(i-io,j-io)];
							u3_d[rel_offset(i,j)] = v3_d[rel_offset(i-io,j-io)];
						}
			}
			#pragma omp barrier
			#pragma omp critical
			{	
				t += 2*nsteps*dt;
			}
			//collect data from each thread and save on main grid
			for (int iy = 0; iy < ndy; ++iy)
				for (int ix = 0; ix < ndx; ++ix) {
					u1(row_offset+ix, col_offset+iy) = u1_d[rel_offset(nghost+ix,nghost+iy)];
					u2(row_offset+ix, col_offset+iy) = u2_d[rel_offset(nghost+ix,nghost+iy)];
					u3(row_offset+ix, col_offset+iy) = u3_d[rel_offset(nghost+ix,nghost+iy)];
				}

			#pragma omp barrier
			#pragma omp flush(done)
		}
		#pragma omp barrier	
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
    real hmin = u1(0,0);
    real hmax = hmin;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            real h = u1(i,j);
            h_sum += h;
            hu_sum += u2(i,j);
            hv_sum += u3(i,j);
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
