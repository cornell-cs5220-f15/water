#include <omp.h>

#define ALLOC alloc_if(1)
#define FREE free_if(1)
#define RETAIN free_if(0)
#define REUSE alloc_if(0)


#ifndef CENTRAL2D_H
#define CENTRAL2D_H

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include "immintrin.h"

#ifndef NWATER
    #define NWATER
    #define NX 200
    #define BLOCKS 4
    #define NBLOCK 50
    #define NPAD 4
    #define NBLOCKALL 58
    #define NSTRIDE 3
#endif


template <class Physics, class Limiter>
class Central2D {
public:
    typedef float real;

    Central2D(real w, real h,     // Domain width / height
              int nx, int ny,     // Number of cells in x/y (without ghosts)
              real cfl = 0.45) :  // Max allowed CFL number
        dx(w/NX), dy(h/NX),
        cfl(cfl) {}

        static const int stride = 16;
        real dtcdx2, dtcdy2;

    // Advance from time 0 to time tfinal
    void run(real tfinal);
    
    // Call f(Uxy, x, y) at each cell center to set initial conditions
    inline __declspec(target (mic)) void init();

    int xsize() const { return NX; }
    int ysize() const { return NX; }

    // Diagnostics
    void solution_check();
    
    real&       operator()(int i, int j) {
        return u1_[i+j*NX];
    }
    
    const real& operator()(int i, int j) const {
        return u1_[i+j*NX];
    }
    
private:
    static constexpr real ghalf = 9.8*0.5;

    const real dx, dy;         // Cell size in x/y
    const real cfl;            // Allowed CFL number
    
    real * u1_ = new real[NX*NX] __attribute__((aligned(32)));            // Solution values
    real * u2_ = new real[NX*NX] __attribute__((aligned(32)));
    real * u3_ = new real[NX*NX] __attribute__((aligned(32)));
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
    for (int iy = 0; iy < NX; ++iy) {
        for (int ix = 0; ix < NX; ++ix){
            real x = (ix+0.5)*dx;
            real y = (iy+0.5)*dy;
            x -= 1;
            y -= 1;
            u1_[ix+iy*NX] = 1.0 + 0.5 * (((x*x + y*y) < 0.25) + 1e-5);
            u2_[ix+iy*NX] = 0;
            u3_[ix+iy*NX] = 0;
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
    const real grav = 9.8f;
    
    printf("Run\n");
    real * su1 = u1_;
    real * su2 = u2_;
    real * su3 = u3_;
    
    printf("Ready\n");
    #pragma offload target(mic) \
        in(dx), in(dy), in(tfinal), \
        inout(su1 : length(NX*NX)), inout(su2 : length(NX*NX)), inout(su3 : length(NX*NX))
    {
        bool done = false;
        
        // Result buffer
        real * res1 = new real[NX*NX];
        real * res2 = new real[NX*NX];
        real * res3 = new real[NX*NX];
        
        real dt = tfinal;
        
        // Begin parallel section, share the current grid state
        #pragma omp parallel shared(dt, tfinal, done, su1, su2, su3, res1, res2, res3)
        {
            //printf("Allocate\n");
            real t = 0.f;
            
            while (!done) {
                // Create private instances
                real * pu1  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pu2  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pu3  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pv1  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pv2  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pv3  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pf1  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pf2  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pf3  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pg1  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pg2  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pg3  = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pux1 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pux2 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pux3 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * puy1 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * puy2 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * puy3 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pfx1 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pfx2 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pfx3 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pgy1 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pgy2 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real * pgy3 = new real[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                
                // Split work by domain
                #pragma omp for
                for (int p = 0; p < BLOCKS * BLOCKS; p++) {
                    // Block x
                    int bx = p / BLOCKS;
                    int bxo = bx * NBLOCK;
                    // Block y
                    int by = p % BLOCKS;
                    int byo = by * NBLOCK;
                    
                    // Copy data to private instances
                    for (int j = 0; j < NBLOCKALL; j++) {
                        for (int i = 0; i < NBLOCKALL; i++) {
                            // Map to indices on grid
                            int wi = (i + bxo - NPAD + NX) % NX;
                            int wj = (j + byo - NPAD + NX) % NX;
                            
                            pu1[j*NBLOCKALL+i] = su1[wj*NX+wi];
                            pu2[j*NBLOCKALL+i] = su2[wj*NX+wi];
                            pu3[j*NBLOCKALL+i] = su3[wj*NX+wi];
                        }
                    }
                    
                    // Compute cx and cy
                    real cx = 0.f, cy = 0.f;
                    for (int iy = 0; iy < NBLOCKALL; ++iy) {
                        for (int ix = 0; ix < NBLOCKALL; ++ix) {
                            real cell_cx, cell_cy;

                            //calculate flux
                            real h  = pu1[ix+iy*NBLOCKALL];
                            real hu = pu2[ix+iy*NBLOCKALL];
                            real hv = pu3[ix+iy*NBLOCKALL];

                            real root_gh = sqrt(grav * h);  // NB: Don't let h go negative!
                            cell_cx = std::abs(hu/h) + root_gh;
                            cell_cy = std::abs(hv/h) + root_gh;
                            cx = std::max(cx, cell_cx);
                            cy = std::max(cy, cell_cy);
                        }
                    }
                    real thisdt = cfl / std::max(cx/dx, cy/dy);
                    
                    // Select min dt > 0
                    #pragma omp critical
                    {
                        if (thisdt > 1e-5 && thisdt < dt) {
                            dt = thisdt;
                        }
                        if (t + (NPAD-3)*2*dt >= tfinal) {
                            dt = (tfinal-t)/((NPAD-3)*2);
                            done = true;
                        }
                    }
                }
                    
                // Use dt
                #pragma omp barrier
                
                // Split work by domain
                #pragma omp for
                for (int p = 0; p < BLOCKS * BLOCKS; p++) {
                    // Block x
                    int bx = p / BLOCKS;
                    int bxo = bx * NBLOCK;
                    // Block y
                    int by = p % BLOCKS;
                    int byo = by * NBLOCK;
                    
                    // Run instances for as many steps as possible
                    for (int step = 0; step < NPAD - 3; step++) {
                        real cx = 1.0e-15;
                        real cy = 1.0e-15;
                        for (int io = 0; io < 2; ++io) {
                            // This is done in the first instance copy
                            // apply_periodic();
                            
                            // compute_fg_speeds
                            for (int iy = 0; iy < NBLOCKALL; ++iy) {
                                for (int ix = 0; ix < NBLOCKALL; ++ix) {
                                    //calculate flux
                                    real h  = pu1[ix+iy*NBLOCKALL];
                                    real hu = pu2[ix+iy*NBLOCKALL];
                                    real hv = pu3[ix+iy*NBLOCKALL];
                                    
                                    pf1[ix+iy*NBLOCKALL] = hu;
                                    pf2[ix+iy*NBLOCKALL] = hu*hu/h + grav*(0.5f)*h*h;
                                    pf3[ix+iy*NBLOCKALL] = hu*hv/h;

                                    pg1[ix+iy*NBLOCKALL] = hv;
                                    pg2[ix+iy*NBLOCKALL] = hu*hv/h;
                                    pg3[ix+iy*NBLOCKALL] = hv*hv/h + grav*(0.5f)*h*h;
                                }
                            }
                            
                            // limited_derivs
                            for (int iy = 1; iy < NBLOCKALL-1; ++iy) {
                                for (int ix = 1; ix < NBLOCKALL-1; ++ix) {
                                    // x derivs
                                    pux1[ix+iy*NBLOCKALL] = Limiter::limdiff(pu1[ix-1+iy*NBLOCKALL], pu1[ix+iy*NBLOCKALL], pu1[ix+1+iy*NBLOCKALL] );
                                    pfx1[ix+iy*NBLOCKALL] = Limiter::limdiff(pf1[ix-1+iy*NBLOCKALL], pf1[ix+iy*NBLOCKALL], pf1[ix+1+iy*NBLOCKALL] );
                                    pux2[ix+iy*NBLOCKALL] = Limiter::limdiff(pu2[ix-1+iy*NBLOCKALL], pu2[ix+iy*NBLOCKALL], pu2[ix+1+iy*NBLOCKALL] );
                                    pfx2[ix+iy*NBLOCKALL] = Limiter::limdiff(pf2[ix-1+iy*NBLOCKALL], pf2[ix+iy*NBLOCKALL], pf2[ix+1+iy*NBLOCKALL] );
                                    pux3[ix+iy*NBLOCKALL] = Limiter::limdiff(pu3[ix-1+iy*NBLOCKALL], pu3[ix+iy*NBLOCKALL], pu3[ix+1+iy*NBLOCKALL] );
                                    pfx3[ix+iy*NBLOCKALL] = Limiter::limdiff(pf3[ix-1+iy*NBLOCKALL], pf3[ix+iy*NBLOCKALL], pf3[ix+1+iy*NBLOCKALL] );

                                    // y derivs
                                    puy1[ix+iy*NBLOCKALL] = Limiter::limdiff( pu1[ix+(iy-1)*NBLOCKALL], pu1[ix+(iy)*NBLOCKALL], pu1[ix+(iy+1)*NBLOCKALL] );
                                    pgy1[ix+iy*NBLOCKALL] = Limiter::limdiff( pg1[ix+(iy-1)*NBLOCKALL], pg1[ix+(iy)*NBLOCKALL], pg1[ix+(iy+1)*NBLOCKALL] );
                                    puy2[ix+iy*NBLOCKALL] = Limiter::limdiff( pu2[ix+(iy-1)*NBLOCKALL], pu2[ix+(iy)*NBLOCKALL], pu2[ix+(iy+1)*NBLOCKALL] );
                                    pgy2[ix+iy*NBLOCKALL] = Limiter::limdiff( pg2[ix+(iy-1)*NBLOCKALL], pg2[ix+(iy)*NBLOCKALL], pg2[ix+(iy+1)*NBLOCKALL] );
                                    puy3[ix+iy*NBLOCKALL] = Limiter::limdiff( pu3[ix+(iy-1)*NBLOCKALL], pu3[ix+(iy)*NBLOCKALL], pu3[ix+(iy+1)*NBLOCKALL] );
                                    pgy3[ix+iy*NBLOCKALL] = Limiter::limdiff( pg3[ix+(iy-1)*NBLOCKALL], pg3[ix+(iy)*NBLOCKALL], pg3[ix+(iy+1)*NBLOCKALL] );
                                }
                            }
                            
                            real dtcdx2 = 0.5 * dt / dx;
                            real dtcdy2 = 0.5 * dt / dy;
                            // Predictor (flux values of f and g at half step)
                            __m512 u1r, fx1r, gy1r, u2r, fx2r, gy2r, u3r, fx3r, gy3r;
                            __m512 ghr, com, f2temp, g3temp;
                            // Predictor (flux values of f and g at half step)
                            for (int iy = 1; iy < NBLOCKALL-1; ++iy) {
                                for (int ix = 1; ix < NBLOCKALL-1; ix++) {
                                    int off = ix+iy*NBLOCKALL;
                                    
                                    real h  = pu1[off] - dtcdx2 * pfx1[off] - dtcdy2 * pgy1[off];
                                    real hu = pu2[off] - dtcdx2 * pfx2[off] - dtcdy2 * pgy2[off];
                                    real hv = pu3[off] - dtcdx2 * pfx3[off] - dtcdy2 * pgy3[off];
                                    
                                    pf1[ix+iy*NBLOCKALL] = hu;
                                    pf2[ix+iy*NBLOCKALL] = hu*hu/h + grav *(0.5)*h*h;
                                    pf3[ix+iy*NBLOCKALL] = hu*hv/h;
                                    
                                    pg1[ix+iy*NBLOCKALL] = hv;
                                    pg2[ix+iy*NBLOCKALL] = hu*hv/h;
                                    pg3[ix+iy*NBLOCKALL] = hv*hv/h + grav *(0.5)*h*h;
                                    
                                    /*
                                    u1r = _mm512_load_ps  (pu1  + off);
                                    fx1r = _mm512_load_ps (pfx1 + off);
                                    gy1r = _mm512_load_ps (pgy1 + off);
                                    u1r = _mm512_fnmadd_ps(fx1r, _mm512_set1_ps(dtcdx2), u1r);
                                    u1r = _mm512_fnmadd_ps(gy1r, _mm512_set1_ps(dtcdy2), u1r);
                                    
                                    u2r = _mm512_load_ps  (pu2  + off);
                                    fx2r = _mm512_load_ps (pfx2 + off);
                                    gy2r = _mm512_load_ps (pgy2 + off);
                                    u2r = _mm512_fnmadd_ps(fx2r, _mm512_set1_ps(dtcdx2), u2r);
                                    u2r = _mm512_fnmadd_ps(gy2r, _mm512_set1_ps(dtcdy2), u2r);
                                    
                                    u3r = _mm512_load_ps  (pu3  + off);
                                    fx3r = _mm512_load_ps (pfx3 + off);
                                    gy3r = _mm512_load_ps (pgy3 + off);
                                    u3r = _mm512_fnmadd_ps(fx3r, _mm512_set1_ps(dtcdx2), u3r);
                                    u3r = _mm512_fnmadd_ps(gy3r, _mm512_set1_ps(dtcdy2), u3r);
                                    
                                    ghr = _mm512_mul_ps(u1r, u1r);
                                    ghr = _mm512_mul_ps(u1r, ghr);
                                    
                                    com = _mm512_mul_ps(u2r, u3r);
                                    com = _mm512_div_ps(com, u1r);
                                    
                                    f2temp = _mm512_div_ps(u2r, u1r);
                                    f2temp = _mm512_fmadd_ps(f2temp, u2r, ghr);
                                    
                                    g3temp = _mm512_div_ps(u3r, u1r);
                                    g3temp = _mm512_fmadd_ps(g3temp, u3r, ghr);
                                    
                                    _mm512_store_ps(pf1 + off, u2r);
                                    _mm512_store_ps(pg1 + off, u3r);
                                    _mm512_store_ps(pf2 + off, f2temp);
                                    _mm512_store_ps(pg2 + off, com);
                                    _mm512_store_ps(pf3 + off, com);
                                    _mm512_store_ps(pg3 + off, g3temp);
                                    */
                                }
                            }
                            
                            // Corrector (finish the step)
                            for (int iy = 1-io; iy < NBLOCKALL-1-io; ++iy) {
                                //unrolling loop x by stride will be inefficient at the lower egdes of the grid
                                for (int ix = 1-io; ix < NBLOCKALL-1-io; ++ix) {
                                    int off0 = ix+iy*NBLOCKALL;
                                    int off1 = (ix+1)+iy*NBLOCKALL;
                                    int off2 = ix+(iy+1)*NBLOCKALL;
                                    int off3 = ix+1+(iy+1)*NBLOCKALL;
                                    
                                    pv1[off0] =
                                        0.2500 * ( pu1 [off0] + pu1 [off1] + pu1 [off2] + pu1 [off3] ) -
                                        0.0625 * ( pux1[off1] - pux1[off0] + pux1[off3] - pux1[off2]   +
                                                   puy1[off2] - puy1[off0] + puy1[off3] - puy1[off1] ) -
                                        dtcdx2 * ( pf1 [off1] - pf1 [off0] + pf1 [off3] - pf1 [off2] ) -
                                        dtcdy2 * ( pg1 [off2] - pg1 [off0] + pg1 [off3] - pg1 [off1] );
                                    
                                    pv2[off0] =
                                        0.2500 * ( pu2 [off0] + pu2 [off1] + pu2 [off2] + pu2 [off3] ) -
                                        0.0625 * ( pux2[off1] - pux2[off0] + pux2[off3] - pux2[off2]   +
                                                   puy2[off2] - puy2[off0] + puy2[off3] - puy2[off1] ) -
                                        dtcdx2 * ( pf2 [off1] - pf2 [off0] + pf2 [off3] - pf2 [off2] ) -
                                        dtcdy2 * ( pg2 [off2] - pg2 [off0] + pg2 [off3] - pg2 [off1] );
                                    
                                    pv3[off0] =
                                        0.2500 * ( pu3 [off0] + pu3 [off1] + pu3 [off2] + pu3 [off3] ) -
                                        0.0625 * ( pux3[off1] - pux3[off0] + pux3[off3] - pux3[off2]   +
                                                   puy3[off2] - puy3[off0] + puy3[off3] - puy3[off1] ) -
                                        dtcdx2 * ( pf3 [off1] - pf3 [off0] + pf3 [off3] - pf3 [off2] ) -
                                        dtcdy2 * ( pg3 [off2] - pg3 [off0] + pg3 [off3] - pg3 [off1] );
                                }
                            }
                            
                             // Copy from v storage back to main grid
                            for (int j = NPAD; j < NBLOCK+NPAD; ++j) {
                                for (int i = NPAD; i < NBLOCK+NPAD; ++i){
                                    pu1[i+j*NBLOCKALL] = pv1[i-io+(j-io)*NBLOCKALL];
                                    pu2[i+j*NBLOCKALL] = pv2[i-io+(j-io)*NBLOCKALL];
                                    pu3[i+j*NBLOCKALL] = pv3[i-io+(j-io)*NBLOCKALL];
                                }
                            }
                            
                            //printf("t %f %f\n", t, dt);
                        }
                    }
                    
                    // Copy data to result grid
                    for (int j = 0; j < NBLOCK; j++) {
                        for (int i = 0; i < NBLOCK; i++) {
                            // Private indices
                            int pi = (i + NPAD);
                            int pj = (j + NPAD);
                            // Map to indices on grid
                            int wi = (i + bxo);
                            int wj = (j + byo);
                            
                            res1[wj*NX+wi] = pu1[pj*NBLOCKALL+pi];
                            res2[wj*NX+wi] = pu2[pj*NBLOCKALL+pi];
                            res3[wj*NX+wi] = pu3[pj*NBLOCKALL+pi];
                        }
                    }
                    
                    t += (NPAD-3)*2*dt;
                }
                
                #pragma omp critical
                {
                    dt = tfinal;
                }
                
                // Wait for all to finish
                #pragma omp barrier
                
                //printf("Copy\n");
                // Copy result grid to original grid
                for (int i = 0; i < NX*NX; i++) {
                    su1[i] = res1[i];
                    su2[i] = res2[i];
                    su3[i] = res3[i];
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
    real hmin = u1_[0];
    real hmax = hmin;
    for (int j = 0; j < NX; ++j)
        for (int i = 0; i < NX; ++i) {
            
            real h = u1_[i+j*NX];
            h_sum += h;
            hu_sum += u2_[i+j*NX];
            hv_sum += u3_[i+j*NX];
            hmax = max(h, hmax);
            hmin = min(h, hmin);
            assert( h > 0 );
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

