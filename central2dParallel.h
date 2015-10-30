#ifndef CENTRAL2D_H
#define CENTRAL2D_H
#include <omp.h>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

template <class Physics, class Limiter>
class Central2D {
public:
  typedef typename Physics::real real;
  typedef typename Physics::vec  vec;

    Central2D(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              real cfl = 0.45) :  // Max allowed CFL number
        nx(nx), ny(ny),
        dx(w/nx), dy(h/ny),
        cfl(cfl), 
        u_ (nx * ny) {}

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
        return u_[offset(i,j)];
    }
    
    const vec& operator()(int i, int j) const {
        return u_[offset(i,j)];
    }
    
private:
    const int nx, ny;          // Number of (non-ghost) cells in x/y
    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number


    std::vector<vec> u_;            // Solution values

    // Array accessor functions
    int offset(int ix, int iy) const { return iy*nx+ix; }
    vec& u(int ix, int iy)    { return u_[offset(ix,iy)]; }


    // Wrapped accessor (periodic BC)
    int ioffset(int ix, int iy) {
        return offset( (ix+nx-nghost) % nx,
                       (iy+ny-nghost) % ny );
    }

    vec& uwrap(int ix, int iy)  { return u_[ioffset(ix,iy)]; }
    // Apply limiter to all components in a vector

    static void limdiff(vec& du, const vec& um, const vec& u0, const vec& up) {
        for (int m = 0; m < du.size(); ++m)
            du[m] = Limiter::limdiff(um[m], u0[m], up[m]);
    }

};

template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::run(real tfinal)
{
        
  int tid,iy,ix,pbx,pby;
  int shift_X,shift_Y;
  int io,iom2;
  const int nsplit = 4;
  const int nxt = nx/nsplit;
  const int nyt = ny/nsplit;
  const int nghost = 3;
  const int nxt_all = nxt + 2*nghost;
  const int nyt_all = nyt + 2*nghost;
  int Nsteps = 1;
  bool done = false;
  real t = 0;
  real dt = 0;


  omp_set_dynamic(0);
  omp_set_num_threads(nsplit*nsplit);
#pragma omp parallel shared(nxt,nyt,nxt_all,nyt_all,nsplit,nghost,done,t,dt,Nsteps) \
  private(tid,shift_X,shift_Y,ix,iy,pbx,pby,io,iom2)
  {

    //// Initializtion of vectors
    std::vector<vec> ut_(nxt_all*nyt_all); 
    std::vector<vec> ft_(nxt_all*nyt_all); 
    std::vector<vec> gt_(nxt_all*nyt_all); 
    std::vector<vec> uxt_(nxt_all*nyt_all); 
    std::vector<vec> uyt_(nxt_all*nyt_all); 
    std::vector<vec> fxt_(nxt_all*nyt_all); 
    std::vector<vec> gyt_(nxt_all*nyt_all); 
    std::vector<vec> vt_(nxt_all*nyt_all); 


    // Defining the thread number and shift_X and shift_Y
    tid = omp_get_thread_num();
    shift_X = (tid%nsplit);
    shift_X = shift_X * nxt;
    shift_Y = (tid/nsplit);
    shift_Y = shift_Y * nyt;

    // Copy the center of the block
    for (iy = 0; iy < nyt; ++iy)
      for (ix = 0; ix < nxt; ++ix) {
	ut_[ix + nghost+ (iy + nghost)*nxt_all] = u_[ix + shift_X+ (iy + shift_Y)*nx];
      }

    while (!done) {
#pragma omp barrier
      if (tid == 0) {
      // Calculation of Time step
	using namespace std;
	real cx = 1.0e-15;
	real cy = 1.0e-15;
	for ( iy = 0; iy < ny; ++iy)
	  for ( ix = 0; ix < nx; ++ix) {
            real cell_cx, cell_cy;
	    Physics::wave_speed(cell_cx, cell_cy, u(ix,iy));
            cx = max(cx, cell_cx);
            cy = max(cy, cell_cy);
	  }
	dt = cfl / std::max(cx/dx, cy/dy);
        if (t >= tfinal) {
          dt = (tfinal-t)/(2);
          Nsteps = 1;
          done = true;
        }
        t = t + 2*Nsteps*dt;
      }
#pragma omp barrier

    // Copy data between right and left boundaries
    for (ix = 0; ix < nxt_all; ++ix)
      for (iy = 0; iy < nghost; ++iy) {
	pbx = (shift_X+ix+nx-nghost)%nx;
	pby = (shift_Y+iy+ny-nghost)%ny;
	ut_[ix + iy*nxt_all] = u_[pbx+pby*nx];

	pby = (shift_Y+iy+nyt+ny)%ny;
	ut_[ix+(nyt+nghost+iy)*nxt_all] = u_[pbx+pby*nx];
      }

    // Copy data between top and bottom boundaries
    for (iy = 0; iy < nyt_all; ++iy)
      for (ix = 0; ix < nghost; ++ix) {
	pby = (shift_Y+iy+ny-nghost)%ny;
	pbx = (shift_X+ix+nx-nghost)%nx;
	ut_[ix + iy*nxt_all] = u_[pbx + pby*nx];
	pbx = (shift_X+ix+nxt+nx)%nx;
	ut_[nxt+nghost+ix+iy*nxt_all] = u_[pbx + pby*nx];
      }


    // Iterative loop to evolve the system
    real dtcdx2 = 0.5 * dt / dx;
    real dtcdy2 = 0.5 * dt / dy;



    for (iom2 = 0; iom2 < 2*Nsteps; iom2 ++) {
      io = iom2%2;
      
      // Flux computation
      for ( iy = 0; iy < nyt_all; ++iy)
        for ( ix = 0; ix < nxt_all; ++ix) {
    	  Physics::flux(ft_[ix+iy*nxt_all], gt_[ix+iy*nxt_all], ut_[ix+iy*nxt_all]);
        }

      // x and y Derivative
      for ( iy = 1; iy < nyt_all-1; ++iy) {
        for ( ix = 1; ix < nxt_all-1; ++ix) {
    	  int fv = ix+iy*nxt_all;
    	  // x derivs
    	  limdiff( uxt_[fv], ut_[fv-1], ut_[fv], ut_[fv+1] );
    	  limdiff( fxt_[fv], ft_[fv-1], ft_[fv], ft_[fv+1] );

    	  // y derives
    	  limdiff( uyt_[fv], ut_[fv-nxt_all], ut_[fv], ut_[fv+nxt_all] );
    	  limdiff( gyt_[fv], gt_[fv-nxt_all], gt_[fv], gt_[fv+nxt_all] );
        }
      }

      // Predictor (flux values of f and g at half step)
      for ( iy = 1; iy < nyt_all-1; ++iy)
        for ( ix = 1; ix < nxt_all-1; ++ix) {
    	  vec uh = ut_[ix+iy*nxt_all];
    	  for (int m = 0; m < uh.size(); ++m) {
    	    uh[m] -= dtcdx2 * fxt_[ix+iy*nxt_all][m];
    	    uh[m] -= dtcdy2 * gyt_[ix+iy*nxt_all][m];
    	  }
    	  Physics::flux(ft_[ix+iy*nxt_all], gt_[ix+iy*nxt_all], uh);
        }

      // Corrector (finish the step)
      for (iy = 1-io; iy < nyt_all-1-io; ++iy) {
        for (ix = 1-io; ix < nxt_all-1-io; ++ix) {
    	  int fv = ix+iy*nxt_all;	  
          for (int m = 0; m < vt_[fv].size(); ++m) {
            vt_[fv][m] =
              0.2500 * ( ut_[fv][m] + ut_[fv+1][m] +
                         ut_[fv+nxt_all][m] + ut_[fv+1+nxt_all][m] ) -
              0.0625 * ( uxt_[fv+1][m] - uxt_[fv][m] +
                         uxt_[fv+1+nxt_all][m] - uxt_[fv+nxt_all][m] +
                         uyt_[fv+nxt_all][m] - uyt_[fv][m] +
                         uyt_[fv+1+nxt_all][m] - uyt_[fv+1][m] ) -
              dtcdx2 * ( ft_[fv+1][m] - ft_[fv][m] +
                         ft_[fv+1+nxt_all][m] - ft_[fv+nxt_all][m] ) -
              dtcdy2 * ( gt_[fv+nxt_all][m] - gt_[fv][m] +
                         gt_[fv+1+nxt_all][m] - gt_[fv+1][m] );
          }
        }
      }
      // Copy from v storage back to main grid
      for (int j = 1; j < nyt_all; ++j){
        for (int i = 1; i < nxt_all; ++i){
    	  ut_[i+j*nxt_all] = vt_[i-io+(j-io)*nxt_all];
        }
      }
    }

    // Write the output to u
    for (int iy = 0; iy < nyt; ++iy)
      for (int ix = 0; ix < nxt; ++ix) {
	u_[ix + shift_X+ (iy + shift_Y)*nx] = ut_[ix + nghost+ (iy + nghost)*nxt_all];
      }
#pragma omp barrier

    }
  }
}

// Initiate u  
template <class Physics, class Limiter>
template <typename F>
void Central2D<Physics, Limiter>::init(F f)
{
   for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix)
            f(u(ix,iy), (ix+0.5)*dx, (iy+0.5)*dy);
    
}



// Diagnostics
template <class Physics, class Limiter>
void Central2D<Physics, Limiter>::solution_check()
{
    using namespace std;
    real h_sum = 0, hu_sum = 0, hv_sum = 0;
    real hmin = u(0,0)[0];
    real hmax = hmin;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
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

#endif /* CENTRAL2D_H*/
