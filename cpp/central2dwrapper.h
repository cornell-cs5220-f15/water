#ifndef CENTRAL2DWRAPPER_H
#define CENTRAL2DWRAPPER_H

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <utility>
#include <omp.h>

#include "central2d.h"

template <class Physics, class Limiter>
class Central2DWrapper {
public:
    typedef typename Physics::real real;
    typedef typename Physics::vec  vec;

    Central2DWrapper(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              real cfl = 0.45) :  // Max allowed CFL number
		dx(w/nx), dy(h/ny),
		nx(nx), ny(ny),
		nx_all(nx + 2*nghost),
		ny_all(ny + 2*nghost),
		cfl(cfl),
		u_ (new std::vector<vec>(nx_all * ny_all)),
		v_ (new std::vector<vec>(nx_all * ny_all)) {}

	~Central2DWrapper() {
		delete(u_);
		delete(v_);
	}

    // Advance from time 0 to time tfinal
	void run(const real tfinal);

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
        return (*u_)[offset(i+nghost,j+nghost)];
    }
    
    const vec& operator()(int i, int j) const {
        return (*u_)[offset(i+nghost,j+nghost)];
    }

private:
	//Large board must use the same number of ghost cells as the small boards
	static constexpr int nghost = Central2D<Physics, Limiter>::nghost;

	static constexpr int nbatch = 2; // Number of steps in a batch before synchronizing

	const real dx, dy;         // Cell size in x/y
    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const int nx_all, ny_all;  // Total cells in x/y (including ghost)
    const real cfl;            // Allowed CFL number

	//Note that these are now stored on the heap so that we can swap
	//the pointers to them between steps, instead of copying from v to u
    std::vector<vec>* u_;            // Solution values
    std::vector<vec>* v_;            // Solution values at next step

    // Array accessor functions

    int offset(int ix, int iy) const { return iy*nx_all+ix; }

    vec& u(int ix, int iy)    { return (*u_)[offset(ix,iy)]; }
    vec& v(int ix, int iy)    { return (*v_)[offset(ix,iy)]; }

    // Wrapped accessor (periodic BC)
    int ioffset(int ix, int iy) {
        return offset( (ix+nx-nghost) % nx + nghost,
                       (iy+ny-nghost) % ny + nghost );
    }

    vec& uwrap(int ix, int iy)  { return (*u_)[ioffset(ix,iy)]; }

	// Compute the wave speeds and choose a timestep, without advancing time
	real compute_current_dt();

	//The only simulation function needed at the wrapper level
    void apply_periodic();

	void check_copyin(Central2D<Physics, Limiter>& wrappedSim, std::vector<vec>* large_u);
	void check_copyout(Central2D<Physics, Limiter>& wrappedSim, std::vector<vec>* large_v);
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
void Central2DWrapper<Physics, Limiter>::init(F f)
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
void Central2DWrapper<Physics, Limiter>::apply_periodic()
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

template <class Physics, class Limiter>
Central2DWrapper<Physics, Limiter>::real Central2DWrapper<Physics, Limiter>::compute_current_dt()
{
    using namespace std;
    real cx = 1.0e-15;
    real cy = 1.0e-15;
    for (int iy = 0; iy < ny_all; ++iy)
        for (int ix = 0; ix < nx_all; ++ix) {
            real cell_cx, cell_cy;
            Physics::wave_speed(cell_cx, cell_cy, u(ix,iy));
            cx = max(cx, cell_cx);
            cy = max(cy, cell_cy);
        }
	real dt = cfl / max(cx/dx, cy/dy);
	return dt;
}

//DEBUG: Only works when there's a single thread!
template <class Physics, class Limiter>
void Central2DWrapper<Physics, Limiter>::check_copyin(Central2D<Physics, Limiter>& wrappedSim, std::vector<vec>* large_u) 
{
	for(int x = 0; x < nx_all; x++) {
		for(int y = 0; y < ny_all; y++) {
			if(wrappedSim.u(x,y) != (*large_u)[offset(x, y)]) {
				printf("Cell (%d, %d) was not equal after copy-in! Central2DWrapper had <%f, %f, %f> but wrapped Central2D had <%f, %f, %f>\n", x, y, (*large_u)[offset(x, y)][0], (*large_u)[offset(x, y)][1], (*large_u)[offset(x, y)][2], wrappedSim.u(x,y)[0], wrappedSim.u(x,y)[1], wrappedSim.u(x,y)[2]);
			}
		}
	}
}

//DEBUG: Only works when there's a single thread!
template <class Physics, class Limiter>
void Central2DWrapper<Physics, Limiter>::check_copyout(Central2D<Physics, Limiter>& wrappedSim, std::vector<vec>* large_v) 
{
	for(int x = 0; x < nx; x++) {
		for(int y = 0; y < ny; y++) {
			if(wrappedSim.u(x+nghost,y+nghost) != (*large_v)[offset(x+nghost, y+nghost)]) {
				printf("Cell (%d, %d) was not equal after copy-out! Central2DWrapper had <%f, %f, %f> but wrapped Central2D had <%f, %f, %f>\n", x+nghost, y+nghost, (*large_v)[offset(x+nghost, y+nghost)][0], (*large_v)[offset(x+nghost, y+nghost)][1], (*large_v)[offset(x+nghost, y+nghost)][2], wrappedSim.u(x+nghost,y+nghost)[0], wrappedSim.u(x+nghost,y+nghost)[1], wrappedSim.u(x+nghost,y+nghost)[2]);
				return;
			}
		}
	}
}

template <class Physics, class Limiter>
void Central2DWrapper<Physics, Limiter>::run(const real tfinal)
{
	typedef Central2D<Physics, Limiter> Central2D_t;
	//OpenMP won't allow you to share member variables, so copy the pointers to u and v into local variables
	std::vector<vec>* large_u = u_;
	std::vector<vec>* large_v = v_;
	//Shared arrays for combining maximum dt values, but can't be initialized until we know the number of threads
	std::vector<real>* local_dts_curr = new std::vector<real>();
	std::vector<real>* local_dts_next = new std::vector<real>();
	//DEBUG
	omp_set_num_threads(1);
	#pragma omp parallel firstprivate(tfinal, large_u, large_v, local_dts_curr,local_dts_next) default(none) 
	{
//		printf("Thread sees Central2DWrapper's nx = %d, ny = %d, nx_all = %d, ny_all = %d \n", nx, ny, nx_all, ny_all);
		const int nthreads = omp_get_num_threads();
		//Now that we know how many threads there are, split the board into
		//approximately that many blocks. It would be nice if we could nicely 
		//factor nthreads into two dimensions instead of just taking the square
		//root and rounding down, but that's beyond the scope of my math knowledge.
		const int blocksperside = (int) std::ceil(std::sqrt(nthreads));
		const int bwidth = nx / blocksperside;
		const int bheight = ny / blocksperside;
		const int nblocksx = nx / bwidth + (nx % bwidth ? 1 : 0);
		const int nblocksy = ny / bheight + (ny % bheight ? 1 : 0);
		const int nblocks = nblocksx * nblocksy;
		//Now we can initialize the local_dts array
		#pragma omp single
		{
			local_dts_curr->resize(nblocks);
			local_dts_next->resize(nblocks);
		}
		//Compute the number of blocks each processor actually gets,
		//which could be >1 since we rounded up the number of blocks
		const int blockspp = nblocks / nthreads;
		const bool one_extra = nblocks % nthreads != 0;
		//Use a Central2D instance for each block; the last thread may get one extra block
		std::vector<Central2D_t> blockSims;
		const int my_numblocks = blockspp + 
			(omp_get_thread_num() == nthreads-1 && one_extra ? 1 : 0);
		//DEBUG
		#pragma omp single 
		{
			printf("nthreads = %d\n", nthreads);
			printf("nblocks = %d\n", nblocks);
			printf("blockspp = %d\n", blockspp);
			printf("one_extra = %d\n", one_extra);
		}
		for(int b = 0; b < my_numblocks; b++) {
			//Blocks are counted in row-major order across the grid
			const int blocknum = b;
			//const int blocknum = omp_get_thread_num() * blockspp + b;
			const int blockrow = blocknum / nblocksx;
			const int blockcol = blocknum % nblocksx;
			//Acutal width and height of this block, accounting for rectangluar ones at the ends
			//note that the block will start at (x,y) = (blockcol * bwidth, blockrow * bheight)
			const int blockwidth = (blockcol * bwidth + bwidth > nx ? nx - blockcol * bwidth : bwidth);
			const int blockheight = (blockrow * bheight + bheight > ny ? ny - blockrow * bheight : bheight);
			//Construct a new Central2D instance for this block. 
			blockSims.push_back(Central2D_t(dx, dy, blockwidth, blockheight, cfl));
		}

		//Before the first iteration, we must compute initial dts. This isn't enough
		//work to parallelize, so just do it in one thread and put the result in every
		//block's local entry. It's ugly, but I don't see any way around it.
		#pragma omp single
		{
			apply_periodic();
			real init_dt = compute_current_dt();
			for(int i = 0; i < nblocks; i++) 
				(*local_dts_curr)[i] = init_dt;
		}
		//Now we can start the actual per-thread computation
		#pragma omp critical
		{
			printf("Thread %d has %d blocks of size %d x %d \n", omp_get_thread_num(), my_numblocks, bwidth, bheight);
		}
		real curtime = 0;
		real dt;
		bool done = false;
		bool first = true;
		while (!done) {
			//Take the minimum of the dt's reported by the previous iteration as the dt for this iteration
			dt = tfinal; //largest value it could be
			for(int i = 0; i < nblocks; i++) {
				if ((*local_dts_curr)[i] < dt)
					dt = (*local_dts_curr)[i];
			}
			//Every thread will reach the same conclusion here, since they have the same dt and curtime
			if (curtime + nbatch*dt >= tfinal) {
				dt = (tfinal-curtime)/nbatch;
				done = true;
			}

			//Hopefully this loop only runs once or twice at each thread
			for(int b = 0; b < my_numblocks; b++) {
				const int blocknum = omp_get_thread_num() * blockspp + b;
				const int blockrow = blocknum / nblocksx;
				const int blockcol = blocknum % nblocksx;
				//Copy this block's data in from u, offsetting pointers by u's ghost cells
				blockSims[b].init_as_subdomain(*large_u, nx_all, 
						nghost + blockcol * bwidth, nghost + blockrow * bheight);
				//Advance two timesteps locally, and put the new dt in dts_next
				(*local_dts_next)[blocknum] = blockSims[b].take_timestep_pair(dt);
				//Copy the results back out to v, so it can happen concurrently with reads from u
				blockSims[b].copy_results_out(*large_v, nx_all,
						nghost + blockcol * bwidth, nghost + blockrow * bheight);
				//If this block is on an edge, also copy out ghost cells
    			if(blockcol == 0) {
    				blockSims[b].copy_vert_ghosts(*large_v, nx_all, nx,
    						nghost + blockrow * bheight, false); 
    			}
    			//This must be checked in sequence, not with ||, because there might be only one column of blocks
    			if(blockcol == nblocksx-1) {
    				blockSims[b].copy_vert_ghosts(*large_v, nx_all, nx,
    						nghost + blockrow * bheight, true); 
    			}
				//Similarly, both might execute if there's only one row of blocks
    			if(blockrow == 0) {
    				blockSims[b].copy_horiz_ghosts(*large_v, nx_all, ny,
    						nghost + blockcol * bwidth, false); 
    			}
    			if(blockrow == nblocksy-1) {
    				blockSims[b].copy_horiz_ghosts(*large_v, nx_all, ny,
    						nghost + blockcol * bwidth, true); 
    			}
				//Corner cases (literally)
				if(blockrow == 0 && blockcol == 0) {
					blockSims[b].copy_corner_ghosts(*large_v, nx_all, 
							ny, nx, Corner::Southeast);
				}
				if(blockrow == 0 && blockcol == nblocksx-1) {
					blockSims[b].copy_corner_ghosts(*large_v, nx_all, 
							ny, nx, Corner::Southwest);
				}
				if(blockrow == nblocksy-1 && blockcol == 0) {
					blockSims[b].copy_corner_ghosts(*large_v, nx_all,
							ny, nx, Corner::Northeast);
				}
				if(blockrow == nblocksy-1 && blockcol == nblocksx-1) {
					blockSims[b].copy_corner_ghosts(*large_v, nx_all,
							ny, nx, Corner::Northwest);
				}
			}

			//Local blocks have now advanced by dt, nbatch times
			curtime += dt*nbatch;

			//Wait for all threads to finish writing out results, then swap "current" and "next" pointers
			#pragma omp barrier
			std::swap(large_u, large_v);
			std::swap(local_dts_curr, local_dts_next);
		} //end while
    } //end omp parallel
	delete(local_dts_curr);
	delete(local_dts_next);
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
void Central2DWrapper<Physics, Limiter>::solution_check()
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

#endif /* CENTRAL2DWRAPPER_H*/

