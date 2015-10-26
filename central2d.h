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
    #define NX 320
    #define BLOCKS 2
    #define NBLOCK 160
    #define NPAD 8
    #define NBLOCKALL 176
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
    
    real u1_[NX*NX];            // Solution values
    real u2_[NX*NX];
    real u3_[NX*NX];

    // Stages of the main algorithm
    inline __declspec(target (mic)) void corrector(int io, real* v, real* u, real* ux, real* uy, real* f, real* g, real dtcdx2, real dtcdy2);
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
            u1_[ix+iy*NX] = 1.0 + 0.5*(x*x + y*y < 0.25+1e-5);
            u2_[ix+iy*NX] = 0;
            u3_[ix+iy*NX] = 0;
        }
    }
}

template <class Physics, class Limiter>
inline __declspec(target (mic)) void Central2D<Physics, Limiter>
::corrector(int io, real v[], real u[], real ux[], real uy[], real f[], real g[], real dtcdx2, real dtcdy2)
{
    #ifdef __MIC__
    
    int ix, iy;
    __m512 u00, u10, u01, u11, ux00, ux10, ux01, ux11;
    __m512 uy00, uy10, uy01, uy11, f00, f01, f10, f11;
    __m512 g00, g01, g10, g11, vr;
    
    // Corrector (finish the step)
    for (int iy = NPAD-io; iy < NBLOCK+NPAD-io; ++iy) {
    //for (int iy = 0; iy < NBLOCKALL; ++iy) {
        //unrolling loop x by stride will be inefficient at the lower egdes of the grid
        for (int ix = NPAD-io; ix < NBLOCK+NPAD-io; ix++) {
            int off0 = ix+iy*NBLOCKALL;
            int off1 = (ix+1)+iy*NBLOCKALL;
            int off2 = ix+(iy+1)*NBLOCKALL;
            int off3 = ix+1+(iy+1)*NBLOCKALL;
            
            v[off0] =
                0.2500 * (  u[off0] +  u[off1] +  u[off2] +  u[off3] ) -
                0.0625 * ( ux[off1] - ux[off0] + ux[off3] - ux[off2]   +
                           uy[off2] - uy[off0] + uy[off3] - uy[off1] ) -
                dtcdx2 * (  f[off1] -  f[off0] +  f[off3] -  f[off2] ) -
                dtcdy2 * (  g[off2] -  g[off0] +  g[off3] -  g[off1] );
            
            /*
            u00 = _mm512_load_ps(u + off0);
            u10 = _mm512_load_ps(u + off1);
            u01 = _mm512_load_ps(u + off2);
            u11 = _mm512_load_ps(u + off3);
            
            ux00 = _mm512_load_ps(ux + off0);
            ux10 = _mm512_load_ps(ux + off1);
            ux01 = _mm512_load_ps(ux + off2);
            ux11 = _mm512_load_ps(ux + off3);
            
            uy00 = _mm512_load_ps(uy + off0);
            uy10 = _mm512_load_ps(uy + off1);
            uy01 = _mm512_load_ps(uy + off2);
            uy11 = _mm512_load_ps(uy + off3);
            
            f00 = _mm512_load_ps(f + off0);
            f10 = _mm512_load_ps(f + off1);
            f01 = _mm512_load_ps(f + off2);
            f11 = _mm512_load_ps(f + off3);
            
            g00 = _mm512_load_ps(g + off0);
            g10 = _mm512_load_ps(g + off1);
            g01 = _mm512_load_ps(g + off2);
            g11 = _mm512_load_ps(g + off3);
            
            u00 = _mm512_add_ps(u00, u10);
            u00 = _mm512_add_ps(u00, u01);
            u00 = _mm512_add_ps(u00, u11);
            u00 = _mm512_mul_ps(u00, _mm512_set1_ps(0.2500f));
            
            ux00 = _mm512_sub_ps(ux10, ux00);
            ux00 = _mm512_add_ps(ux00, ux11);
            ux00 = _mm512_sub_ps(ux00, ux01);
            ux00 = _mm512_add_ps(ux00, uy01);
            ux00 = _mm512_sub_ps(ux00, uy00);
            ux00 = _mm512_add_ps(ux00, uy11);
            ux00 = _mm512_sub_ps(ux00, uy10);
            ux00 = _mm512_mul_ps(ux00, _mm512_set1_ps(0.0625f));
            
            f00 = _mm512_sub_ps(f10, f00);
            f00 = _mm512_add_ps(f00, f11);
            f00 = _mm512_sub_ps(f00, f01);
            f00 = _mm512_mul_ps(f00, _mm512_set1_ps(dtcdx2));
            
            g00 = _mm512_sub_ps(g01, g00);
            g00 = _mm512_add_ps(g00, g11);
            g00 = _mm512_sub_ps(g00, g10);
            g00 = _mm512_mul_ps(g00, _mm512_set1_ps(dtcdy2));
            
            vr = _mm512_sub_ps(u00, ux00);
            vr = _mm512_sub_ps(vr, f00);
            vr = _mm512_sub_ps(vr, g00);
            
            _mm512_store_ps(v, vr);
            */
        }
    }
    #endif
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
    printf("Run\n");
    float * su1 = u1_;
    float * su2 = u2_;
    float * su3 = u3_;
    printf("Ready\n");
    #pragma offload target(mic) \
        in(dx), in(dy), \
        in(su1 : length(NX*NX)), in(su2 : length(NX*NX)), in(su3 : length(NX*NX))
    {
        printf("Offload\n");
        bool done = false;
        real currtime = 0;
        real endtime = tfinal;
        printf("Loop\n");
        while (!done) {
            printf("Allocate\n");
            
            // OpenMP sucks and doesn't allow members to be shared
            real res1[NX*NX];
            real res2[NX*NX];
            real res3[NX*NX];
            
            // Begin parallel section, share the current grid state
            #pragma omp parallel shared(currtime, endtime, su1, su2, su3, res1, res2, res3)
            {
                // Copy the current time
                real t = currtime;
                
                // Create private instances
                real pu1 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pu2 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pu3 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pv1 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pv2 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pv3 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pf1 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pf2 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pf3 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pg1 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pg2 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pg3 [NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pux1[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pux2[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pux3[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real puy1[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real puy2[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real puy3[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pfx1[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pfx2[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pfx3[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pgy1[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pgy2[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                real pgy3[NBLOCKALL*NBLOCKALL] __attribute__((aligned(32)));
                
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
                            int wi = (i + bxo - NPAD) % NX;
                            int wj = (j + byo - NPAD) % NX;
                            
                            pu1[j*NBLOCKALL+i] = su1[wj*NX+wj];
                            pu2[j*NBLOCKALL+i] = su2[wj*NX+wj];
                            pu3[j*NBLOCKALL+i] = su3[wj*NX+wj];
                        }
                    }
                    
                    // Run instances for as many steps as possible
                    for (int step = 0; step < NPAD - 3 && !done; step++) {
                        real dt;
                        real cx = 1.0e-15;
                        real cy = 1.0e-15;
                        for (int io = 0; io < 2; ++io) {
                            // This is done in the first instance copy
                            // apply_periodic();
                            
                            // compute_fg_speeds
                            const real grav = 9.8;
                            for (int iy = 0; iy < NBLOCKALL; ++iy) {
                                for (int ix = 0; ix < NBLOCKALL; ++ix) {
                                    real cell_cx, cell_cy;

                                    //calculate flux
                                    real h  = pu1[ix+iy*NBLOCKALL];
                                    real hu = pu2[ix+iy*NBLOCKALL];
                                    real hv = pu3[ix+iy*NBLOCKALL];
                                    pf1[ix+iy*NBLOCKALL] = hu;
                                    pf2[ix+iy*NBLOCKALL] = hu*hu/h + grav *(0.5)*h*h;
                                    pf3[ix+iy*NBLOCKALL] = hu*hv/h;

                                    pg1[ix+iy*NBLOCKALL] = hv;
                                    pg2[ix+iy*NBLOCKALL] = hu*hv/h;
                                    pg3[ix+iy*NBLOCKALL] = hv*hv/h + grav *(0.5)*h*h;

                                    real root_gh = sqrt(grav * h);  // NB: Don't let h go negative!
                                    cell_cx = std::abs(hu/h) + root_gh;
                                    cell_cy = std::abs(hv/h) + root_gh;
                                    cx = std::max(cx, cell_cx);
                                    cy = std::max(cy, cell_cy);
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
                            
                            // Time step
                            if (io == 0) {
                                dt = cfl / std::max(cx/dx, cy/dy);
                                if (t + 2*dt >= endtime) {
                                    dt = (endtime-t)/2;
                                    done = true;
                                }
                            }
                            
                            // compute_fg_speeds
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
                            corrector(io, pv1, pu1, pux1, puy1, pf1, pg1, dtcdx2, dtcdy2);
                            corrector(io, pv2, pu2, pux2, puy2, pf2, pg2, dtcdx2, dtcdy2);
                            corrector(io, pv3, pu3, pux3, puy3, pf3, pg3, dtcdx2, dtcdy2);
                            
                             // Copy from v storage back to main grid
                            for (int j = NPAD; j < NBLOCK+NPAD; ++j){
                                for (int i = NPAD; i < NBLOCK+NPAD; ++i){
                                    pu1[i+j*NBLOCKALL] = pv1[i-io+(j-io)*NBLOCKALL];
                                    pu2[i+j*NBLOCKALL] = pv2[i-io+(j-io)*NBLOCKALL];
                                    pu3[i+j*NBLOCKALL] = pv3[i-io+(j-io)*NBLOCKALL];
                                }
                            }
                            t += dt;
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
                            
                            res1[wj*NX+wj] = pu1[pj*NBLOCKALL+pi];
                            res2[wj*NX+wj] = pu2[pj*NBLOCKALL+pi];
                            res3[wj*NX+wj] = pu3[pj*NBLOCKALL+pi];
                        }
                    }
                }
                
                #pragma omp critical
                currtime = t;
                
                // Wait for all to finish
                #pragma omp barrier
            }
            
            printf("Copy\n");
            // Copy result grid to original grid
            for (int i = 0; i < NX*NX; i++) {
                su1[i] = res1[i];
                su2[i] = res2[i];
                su3[i] = res3[i];
            }
            
            done = true;
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

