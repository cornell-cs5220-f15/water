#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#include <cstdio>
#include <cmath>
#include <omp.h>
#include <cassert>
#include <vector>

const int num_thread = 5;
// needs to divide nx and ny evenly
const int SUBDOMAIN_SIZE = 128;


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
class  Central2D {
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
    
private:
    static constexpr int nghost = 3;   // Number of ghost cells
    static const int SIZE_WITH_GHOST = SUBDOMAIN_SIZE + 2*nghost;

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
    
    vec& castIndex(std::vector<vec>* vecArg, int ix, int iy) {
        return (vec&) vecArg[iy*SIZE_WITH_GHOST + ix];
    }

    // Apply limiter to all components in a vector
    static void limdiff(vec& du, const vec& um, const vec& u0, const vec& up) {
        for (int m = 0; m < du.size(); ++m)
            du[m] = Limiter::limdiff(um[m], u0[m], up[m]);
    }

    // Stages of the main algorithm
    void apply_periodic(std::vector<vec>* U, std::vector<vec>* upperL_u,
        std::vector<vec>* upper_u, std::vector<vec>* upperR_u,
        std::vector<vec>* left_u, std::vector<vec>* right_u,
        std::vector<vec>* lowerL_u, std::vector<vec>* lower_u,
        std::vector<vec>* lowerR_u);
    void compute_fg_speeds(std::vector<vec>* U, std::vector<vec>* fluxX,
        std::vector<vec>* fluxY, real& cx, real& cy);
    void limited_derivs(std::vector<vec>* U, std::vector<vec>* fluxX,
        std::vector<vec>* fluxY, std::vector<vec>* UX, std::vector<vec>* UY,
        std::vector<vec>* FX, std::vector<vec>* GY);
    void compute_step(std::vector<vec>* U, std::vector<vec>* fluxX,
        std::vector<vec>* fluxY, std::vector<vec>* UX, std::vector<vec>* UY,
        std::vector<vec>* FX, std::vector<vec>* GY, int io, real dt);

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
	#pragma omp parallel for
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            f(u(nghost+ix,nghost+iy), (ix+0.5)*dx, (iy+0.5)*dy);
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
void Central2D<Physics, Limiter>::apply_periodic(std::vector<vec>* U,
    std::vector<vec>* upperL_u, std::vector<vec>* upper_u,
    std::vector<vec>* upperR_u, std::vector<vec>* left_u,
    std::vector<vec>* right_u, std::vector<vec>* lowerL_u,
    std::vector<vec>* lower_u, std::vector<vec>* lowerR_u) {
        
    // indices 0 to SUBDOMAIN_SIZE are regular cells
    // indices SUBDOMAIN_SIZE to (SUBDOMAIN_SIZE + 2 * nghost = SIZE_WITH_GHOST) are ghost cells
    // this is different from the starter code
    
	#pragma omp parallel for
    // Copy data between right and left boundaries
    for (int iy = 0; iy < SUBDOMAIN_SIZE; ++iy) {
        int yOffset = iy * SIZE_WITH_GHOST;
        for (int ix = 0; ix < nghost; ++ix) {
            int xOffset = ix + SUBDOMAIN_SIZE;
            
            U[yOffset + xOffset] = right_u[yOffset + ix];
            U[yOffset + xOffset + nghost] = left_u[yOffset + xOffset - nghost];
        }
    }

	#pragma omp parallel for
    // Copy data between top and bottom boundaries
    for (int ix = 0; ix < SUBDOMAIN_SIZE; ++ix)
        for (int iy = 0; iy < nghost; ++iy) {
            int yOffset = (iy + SUBDOMAIN_SIZE) * SIZE_WITH_GHOST;
            
            U[yOffset + ix] = lower_u[iy * SIZE_WITH_GHOST + ix];
            U[yOffset + ix + nghost*SIZE_WITH_GHOST] = upper_u[ix + yOffset - nghost*SIZE_WITH_GHOST];
        }
    
    // copy data from corners
    for( int ix = 0; ix < nghost; ++ix ) {
        int xOffset = SUBDOMAIN_SIZE + ix;
        for( int iy = 0; iy < nghost; ++iy ) {
            int yOffset = (SUBDOMAIN_SIZE + iy) * SIZE_WITH_GHOST;
            
            U[yOffset + xOffset] = lowerR_u[iy*SIZE_WITH_GHOST + ix];
            U[yOffset + xOffset + nghost] = lowerL_u[iy*SIZE_WITH_GHOST + xOffset - nghost];
            U[yOffset + nghost*SIZE_WITH_GHOST + xOffset] = upperR_u[(SUBDOMAIN_SIZE + iy - nghost)*SIZE_WITH_GHOST + ix];
            U[yOffset + nghost*SIZE_WITH_GHOST + xOffset + nghost] = upperL_u[(SUBDOMAIN_SIZE + iy - nghost)*SIZE_WITH_GHOST + xOffset - nghost];
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
void Central2D<Physics, Limiter>::compute_fg_speeds(std::vector<vec>* U,
    std::vector<vec>* fluxX, std::vector<vec>* fluxY, real& cx_, real& cy_) {
        
    using namespace std;
    real cx = 1.0e-15;
    real cy = 1.0e-15;
    
    // was previously looping through everything including ghost cells?
    // if broken, change loop bounds to include ghost cells
	#pragma omp parallel for
    for (int iy = 0; iy < SIZE_WITH_GHOST; ++iy)
        for (int ix = 0; ix < SIZE_WITH_GHOST; ++ix) {
            real cell_cx, cell_cy;
            Physics::flux(castIndex(fluxX, ix, iy), castIndex(fluxY, ix, iy), castIndex(U, ix, iy));
            Physics::wave_speed(cell_cx, cell_cy, castIndex(U, ix, iy));
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
void Central2D<Physics, Limiter>::limited_derivs(std::vector<vec>* U,
    std::vector<vec>* fluxX, std::vector<vec>* fluxY, std::vector<vec>* UX,
    std::vector<vec>* UY, std::vector<vec>* FX, std::vector<vec>* GY) {
        
    // if broken, change loop bounds to 1 to SIZE_WITH_GHOST - 1
	// this is slow
	#pragma omp parallel for
    for (int y = 0; y < SIZE_WITH_GHOST - 2; ++y) {
        int iy = (y - nghost + 1) % SIZE_WITH_GHOST;
        for (int x = 0; x < SIZE_WITH_GHOST - 2; ++x) {
            int ix = (x - nghost + 1) % SIZE_WITH_GHOST;
            
            // x derivs
            limdiff( castIndex(UX, ix, iy), castIndex(U, ix-1, iy), castIndex(U, ix, iy), castIndex(U, ix+1, iy) );
            limdiff( castIndex(FX, ix, iy), castIndex(fluxX, ix-1, iy), castIndex(fluxX, ix, iy), castIndex(fluxX, ix+1, iy) );

            // y derivs
            limdiff( castIndex(UY, ix, iy), castIndex(U, ix, iy-1), castIndex(U, ix, iy), castIndex(U, ix, iy+1) );
            limdiff( castIndex(GY, ix, iy), castIndex(fluxY, ix, iy-1), castIndex(fluxY, ix, iy), castIndex(fluxY, ix, iy+1) );
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
void Central2D<Physics, Limiter>::compute_step(std::vector<vec>* U,
    std::vector<vec>* fluxX, std::vector<vec>* fluxY, std::vector<vec>* UX,
    std::vector<vec>* UY, std::vector<vec>* FX, std::vector<vec>* GY,
    int io, real dt) {
        
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;

    // if broken, revisit loop bounds 1 to (ny/x_all-1)
    // Predictor (flux values of f and g at half step)
	#pragma omp parallel for
    for (int y = 0; y < SIZE_WITH_GHOST - 2; ++y) {
        int iy = (y - nghost + 1) % SIZE_WITH_GHOST;
        for (int x = 0; x < SIZE_WITH_GHOST - 2; ++x) {
            int ix = (x - nghost + 1) % SIZE_WITH_GHOST;
            vec uh = castIndex(U, ix, iy);
            for (int m = 0; m < uh.size(); ++m) {
                uh[m] -= dtcdx2 * castIndex(FX, ix, iy)[m];
                uh[m] -= dtcdy2 * castIndex(GY, ix, iy)[m];
            }
            Physics::flux(castIndex(fluxX, ix, iy), castIndex(fluxY, ix, iy), uh);
        }
    }
    
    std::vector<vec>* V = (std::vector<vec>*) malloc(SIZE_WITH_GHOST * SIZE_WITH_GHOST * sizeof(std::vector<vec>));

    // if broken, revisit loop bounds (nghost-io) to (ny+nghost-io)
    // Corrector (finish the step)
	#pragma omp parallel for
    for (int y = 0; y < SUBDOMAIN_SIZE; ++y) {
        int iy = (y - io) % SIZE_WITH_GHOST;
        for (int x = 0; x < SUBDOMAIN_SIZE; ++x) {
            int ix = (x - io) % SIZE_WITH_GHOST;
            for (int m = 0; m < v(ix,iy).size(); ++m) {
                castIndex(V, ix, iy)[m] =
                    0.2500 * ( castIndex(U, ix, iy)[m] + castIndex(U, ix+1, iy)[m] +
                               castIndex(U, ix, iy+1)[m] + castIndex(U, ix+1, iy+1)[m] ) -
                    0.0625 * ( castIndex(UX, ix+1, iy)[m] - castIndex(UX, ix, iy)[m] +
                               castIndex(UX, ix+1, iy+1)[m] - castIndex(UX, ix, iy+1)[m] +
                               castIndex(UY, ix, iy+1)[m] - castIndex(UY, ix, iy)[m] +
                               castIndex(UY, ix+1, iy+1)[m] - castIndex(UY, ix+1, iy)[m] ) -
                    dtcdx2 * ( castIndex(fluxX, ix+1, iy)[m] - castIndex(fluxX, ix, iy)[m] +
                               castIndex(fluxX, ix+1, iy+1)[m] - castIndex(fluxX, ix, iy+1)[m] ) -
                    dtcdy2 * ( castIndex(fluxY, ix, iy+1)[m] - castIndex(fluxY, ix, iy)[m] +
                               castIndex(fluxY, ix+1, iy+1)[m] - castIndex(fluxY, ix+1, iy)[m] );
            }
        }
    }

    // if broken, loop bounds were nghost to (nx/y + nghost)
    // Copy from v storage back to main grid
	#pragma omp parallel for
    for (int iy = 0; iy < SUBDOMAIN_SIZE; ++iy){
        for (int ix = 0; ix < SUBDOMAIN_SIZE; ++ix){
            int offset = (iy-io)*SIZE_WITH_GHOST + ix-io;
            U[iy*SIZE_WITH_GHOST + ix] = V[offset];
        }
    }
}

int computeBlockOffset(int x, int y, int numBlocksX) {
    return y*numBlocksX + x;
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
    using namespace std;
    
    // divide domain into blocks
    const int NUM_BLOCKS_X = (nx / SUBDOMAIN_SIZE) + (nx % SUBDOMAIN_SIZE ? 1 : 0);
    const int NUM_BLOCKS_Y = (ny / SUBDOMAIN_SIZE) + (ny % SUBDOMAIN_SIZE ? 1 : 0);
    
    // points at each subdomain block's start memory address
    std::vector<vec>** subDomainPointers_u = (std::vector<vec>**) malloc(NUM_BLOCKS_X * NUM_BLOCKS_Y * sizeof(std::vector<vec>*));
    
    // malloc each subdomain's memory (including space for ghost cells)
    for( int j=0; j < NUM_BLOCKS_Y; ++j ) {
        for( int i=0; i < NUM_BLOCKS_X; ++i ) {
            std::vector<vec>* subdomain_u = (std::vector<vec>*) malloc(SIZE_WITH_GHOST * SIZE_WITH_GHOST * sizeof(std::vector<vec>));
            
            subDomainPointers_u[j*NUM_BLOCKS_X + i] = subdomain_u;
            
            // to read from original grid, offset to block start location
            int xBlockOffset = i * SUBDOMAIN_SIZE;
            int yBlockOffset = j * SUBDOMAIN_SIZE;
            //int blockOffset = yBlockOffset*nx + xBlockOffset;
            
            // copy initial values (0,0) to (SUBDOMAIN_SIZE, SUBDOMAIN_SIZE)
            // ignore ghost cells because they are copied later in apply_periodic()
            // ignore fluxes because they are computed each step
            for( int y=0; y < SUBDOMAIN_SIZE; ++y ) {
                for( int x=0; x < SUBDOMAIN_SIZE; ++x ) {
                    int xCoord = xBlockOffset + x;
                    int yCoord = yBlockOffset + y;
                    
                    // if outside domain, copy zero
                    if(xCoord < nx && yCoord < ny) {
                        vec& U = u(xCoord, yCoord);
                        castIndex(subdomain_u, y, x)[0] = U[0];
                        castIndex(subdomain_u, y, x)[1] = U[1];
                        castIndex(subdomain_u, y, x)[2] = U[2];
                        
                    } else {
                        castIndex(subdomain_u, y, x)[0] = 0.0;
                        castIndex(subdomain_u, y, x)[1] = 0.0;
                        castIndex(subdomain_u, y, x)[2] = 0.0;
                    }
                }
            }
        }
    }
    
    
    bool done = false;
    real t = 0;
    while (!done) {
        real dt;
        
        // thread start
        int myXBlock = 1;
        int myYBlock = 1;
        std::vector<vec>* myU = subDomainPointers_u[computeBlockOffset(myXBlock, myYBlock, NUM_BLOCKS_X)];
        
        // these are generated during the loop, no need to pre-allocate
        std::vector<vec>* myFluxX = (std::vector<vec>*) malloc(SIZE_WITH_GHOST * SIZE_WITH_GHOST * sizeof(std::vector<vec>));
        std::vector<vec>* myFluxY = (std::vector<vec>*) malloc(SIZE_WITH_GHOST * SIZE_WITH_GHOST * sizeof(std::vector<vec>));
        std::vector<vec>* myUX = (std::vector<vec>*) malloc(SIZE_WITH_GHOST * SIZE_WITH_GHOST * sizeof(std::vector<vec>));
        std::vector<vec>* myUY = (std::vector<vec>*) malloc(SIZE_WITH_GHOST * SIZE_WITH_GHOST * sizeof(std::vector<vec>));
        std::vector<vec>* myFX = (std::vector<vec>*) malloc(SIZE_WITH_GHOST * SIZE_WITH_GHOST * sizeof(std::vector<vec>));
        std::vector<vec>* myGY = (std::vector<vec>*) malloc(SIZE_WITH_GHOST * SIZE_WITH_GHOST * sizeof(std::vector<vec>));
        
        for (int io = 0; io < 1; ++io) {
            
            real cx, cy;
            
            // grab adjacent block arrays, needed for ghost cell copying
            int upBlock = (myYBlock - 1) % NUM_BLOCKS_Y;
            int leftBlock = (myXBlock - 1) % NUM_BLOCKS_X;
            int rightBlock = (myXBlock + 1) % NUM_BLOCKS_X;
            int downBlock = (myYBlock + 1) % NUM_BLOCKS_Y;
            std::vector<vec>* upperL_u = subDomainPointers_u[computeBlockOffset(leftBlock, upBlock, NUM_BLOCKS_X)];
            std::vector<vec>* upper_u = subDomainPointers_u[computeBlockOffset(myXBlock, upBlock, NUM_BLOCKS_X)];
            std::vector<vec>* upperR_u = subDomainPointers_u[computeBlockOffset(rightBlock, upBlock, NUM_BLOCKS_X)];
            std::vector<vec>* left_u = subDomainPointers_u[computeBlockOffset(leftBlock, myYBlock, NUM_BLOCKS_X)];
            std::vector<vec>* right_u = subDomainPointers_u[computeBlockOffset(rightBlock, myYBlock, NUM_BLOCKS_X)];
            std::vector<vec>* lowerL_u = subDomainPointers_u[computeBlockOffset(leftBlock, downBlock, NUM_BLOCKS_X)];
            std::vector<vec>* lower_u = subDomainPointers_u[computeBlockOffset(myXBlock, downBlock, NUM_BLOCKS_X)];
            std::vector<vec>* lowerR_u = subDomainPointers_u[computeBlockOffset(rightBlock, downBlock, NUM_BLOCKS_X)];
            // TODO: handle case where nx % SUBDOMAIN_SIZE > 0
            // this is difficult when copying ghost cells
            //
            
            // TODO: optimize with static to avoid so many arguments (fatal bad_alloc error)
            //apply_periodic(myU, upperL_u, upper_u, upperR_u, left_u,
            //               right_u, lowerL_u, lower_u, lowerR_u);
            
            
            compute_fg_speeds(myU, myFluxX, myFluxY, cx, cy);
            // TODO: place barrier here to sync max speed
            // BARRIER
            
            // limited_derivs() is seg faulting, fatal
            //limited_derivs(myU, myFluxX, myFluxY, myUX, myUY, myFX, myGY);
            /**
            if (io == 0) {
                dt = cfl / std::max(cx/dx, cy/dy);
                if (t + 2*dt >= tfinal) {
                    dt = (tfinal-t)/2;
                    done = true;
                }
            }**/
            
            // compute_step() is seg faulting, fatal
            compute_step(myU, myFluxX, myFluxY, myUX, myUY, myFX, myGY, io, dt);
            t += dt;
        }
        done = true;
        
        free(myFluxX);
        free(myFluxY);
        free(myUY);
        free(myUX);
        free(myFX);
        free(myGY);
    }
    
    
    // copy subdomains back into master array
    for( int j=0; j < NUM_BLOCKS_Y; ++j ) {
        for( int i=0; i < NUM_BLOCKS_X; ++i ) {
            std::vector<vec>* myU = subDomainPointers_u[j*NUM_BLOCKS_X + i];
            
            // to write to original grid, offset to block start location
            int xBlockOffset = i * SUBDOMAIN_SIZE;
            int yBlockOffset = j * SUBDOMAIN_SIZE;
            //int blockOffset = yBlockOffset*nx + xBlockOffset;
            
            for( int y=0; y < SUBDOMAIN_SIZE; ++y ) {
                for( int x=0; x < SUBDOMAIN_SIZE; ++x ) {
                    int xCoord = xBlockOffset + x;
                    int yCoord = yBlockOffset + y;
                    
                    if(xCoord < nx && yCoord < ny) {
                        vec& U = u(xCoord, yCoord);
                        U[0] = castIndex(myU, y, x)[0];
                        U[1] = castIndex(myU, y, x)[1];
                        U[2] = castIndex(myU, y, x)[2];
                        
                        /**
                        int b = 1;
                        b = b && U[0] == castIndex(myU, y, x)[0];
                        b = b && U[1] == castIndex(myU, y, x)[1];
                        b = b && U[2] == castIndex(myU, y, x)[2];
                        
                        if(b == 0) {
                            printf("false!\n");
                        }**/
                        
                    }
                }
            }
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
	#pragma omp parallel for
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
    /**printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n",
           h_sum, hu_sum, hv_sum, hmin, hmax);**/
}

//ldoc off
#endif /* CENTRAL2D_H*/
