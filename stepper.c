#include "stepper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <immintrin.h>
#include <omp.h>

//ldoc on
/**
 * ## Implementation
 *
 * ### Structure allocation
 */

central2d_t* __attribute__((target(mic))) central2d_init(float w, float h, int nx, int ny,
                            int nfield,
                            float cfl, int b)
{
    int ng = 4*b;

    central2d_t* sim = (central2d_t*) _mm_malloc(sizeof(central2d_t), 64);
    sim->nx = nx;
    sim->ny = ny;
    sim->ng = ng;
    sim->nfield = nfield;
    sim->dx = w/nx;
    sim->dy = h/ny;
    sim->cfl = cfl;

    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    int nc = nx_all * ny_all;
    int N  = nfield * nc;
    sim->u  = (float*) _mm_malloc((4*N + 6*nx_all)* sizeof(float), 64);
    sim->v  = sim->u +   N;
    sim->f  = sim->u + 2*N;
    sim->g  = sim->u + 3*N;
    sim->scratch = sim->u + 4*N;

    return sim;
}


void __attribute__((target(mic))) central2d_free(central2d_t* sim)
{
    _mm_free(sim->u);
    _mm_free(sim);
}

static const float __attribute__((target(mic))) g = 9.8;


static
void __attribute__((target(mic))) shallow2dv_flux(float* restrict fh,
                     float* restrict fhu,
                     float* restrict fhv,
                     float* restrict gh,
                     float* restrict ghu,
                     float* restrict ghv,
                     const float* restrict h,
                     const float* restrict hu,
                     const float* restrict hv,
                     float g,
                     int ncell)
{
    memcpy(fh, hu, ncell * sizeof(float));
    memcpy(gh, hv, ncell * sizeof(float));
    // #pragma vector aligned
    for (int i = 0; i < ncell; ++i) {
        float hi = h[i], hui = hu[i], hvi = hv[i];
        float inv_h = 1/hi;
        fhu[i] = hui*hui*inv_h + (0.5f*g)*hi*hi;
        fhv[i] = hui*hvi*inv_h;
        ghu[i] = hui*hvi*inv_h;
        ghv[i] = hvi*hvi*inv_h + (0.5f*g)*hi*hi;
    }
}


static
void __attribute__((target(mic))) shallow2dv_speed(float* restrict cxy,
                      const float* restrict h,
                      const float* restrict hu,
                      const float* restrict hv,
                      float g,
                      int ncell)
{
    float cx = cxy[0];
    float cy = cxy[1];
    // #pragma vector aligned
    for (int i = 0; i < ncell; ++i) {
        float hi = h[i];
        float inv_hi = 1.0f/h[i];
        float root_gh = sqrtf(g * hi);
        float cxi = fabsf(hu[i] * inv_hi) + root_gh;
        float cyi = fabsf(hv[i] * inv_hi) + root_gh;
        if (cx < cxi) cx = cxi;
        if (cy < cyi) cy = cyi;
    }
    cxy[0] = cx;
    cxy[1] = cy;
}


void __attribute__((target(mic))) shallow2d_flux(float* FU, float* GU, const float* U,
                    int ncell, int field_stride)
{
    shallow2dv_flux(FU, FU+field_stride, FU+2*field_stride,
                    GU, GU+field_stride, GU+2*field_stride,
                    U,  U +field_stride, U +2*field_stride,
                    g, ncell);
}


void __attribute__((target(mic))) shallow2d_speed(float* cxy, const float* U,
                     int ncell, int field_stride)
{
    shallow2dv_speed(cxy, U, U+field_stride, U+2*field_stride, g, ncell);
}



int __attribute__((target(mic))) central2d_offset(int nx, int ny, int ng, int k, int ix, int iy)
{
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    return (k*ny_all+(ng+iy))*nx_all+(ng+ix);
}

//offsets wrt to 0,0 absolute memory position
int __attribute__((target(mic))) central2d_offset_absolute(int nx, int ny, int ng, int k, int ix, int iy)
{
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    return (k*ny_all+(iy))*nx_all+(ix);
}

/**
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

static inline
void __attribute__((target(mic))) copy_subgrid(float* restrict dst,
                  const float* restrict src,
                  int nx, int ny, int stride)
{
    // #pragma vector aligned
    for (int iy = 0; iy < ny; ++iy)
        // #pragma vector aligned
        for (int ix = 0; ix < nx; ++ix)
            dst[iy*stride+ix] = src[iy*stride+ix];
}

void __attribute__((target(mic))) central2d_periodic(float* restrict u,
                        int nx, int ny, int ng, int nfield)
{
    // Stride and number per field
    int s = nx + 2*ng;
    int field_stride = (ny+2*ng)*s;

    // Offsets of left, right, top, and bottom data blocks and ghost blocks
    int l = nx,   lg = 0;
    int r = ng,   rg = nx+ng;
    int b = ny*s, bg = 0;
    int t = ng*s, tg = (nx+ng)*s;

    // Copy data into ghost cells on each side
    // #pragma vector aligned
    for (int k = 0; k < nfield; ++k) {
        float* uk = u + k*field_stride;
        copy_subgrid(uk+lg, uk+l, ng, ny+2*ng, s);
        copy_subgrid(uk+rg, uk+r, ng, ny+2*ng, s);
        copy_subgrid(uk+tg, uk+t, nx+2*ng, ng, s);
        copy_subgrid(uk+bg, uk+b, nx+2*ng, ng, s);
    }
}


/**
 * ### Derivatives with limiters
 *
 * In order to advance the time step, we also need to estimate
 * derivatives of the fluxes and the solution values at each cell.
 * In order to maintain stability, we apply a limiter here.
 *
 * The minmod limiter *looks* like it should be expensive to computer,
 * since superficially it seems to require a number of branches.
 * We do something a little tricky, getting rid of the condition
 * on the sign of the arguments using the `copysign` instruction.
 * If the compiler does the "right" thing with `max` and `min`
 * for floating point arguments (translating them to branch-free
 * intrinsic operations), this implementation should be relatively fast.
 */


// Branch-free computation of minmod of two numbers times 2s
static inline
float __attribute__((target(mic))) xmin2s(float s, float a, float b) {
    float sa = copysignf(s, a);
    float sb = copysignf(s, b);
    float abs_a = fabsf(a);
    float abs_b = fabsf(b);
    float min_abs = (abs_a < abs_b ? abs_a : abs_b);
    return (sa+sb) * min_abs;
}


// Limited combined slope estimate
static inline
float __attribute__((target(mic))) limdiff(float um, float u0, float up) {
    const float theta = 2.0;
    const float quarter = 0.25;
    float du1 = u0-um;   // Difference to left
    float du2 = up-u0;   // Difference to right
    float duc = up-um;   // Twice centered difference
    return xmin2s( quarter, xmin2s(theta, du1, du2), duc );
}


// Compute limited derivs
static inline
void __attribute__((target(mic))) limited_deriv1(float* restrict du,
                    const float* restrict u,
                    int ncell)
{
    // #pragma vector aligned
    for (int i = 0; i < ncell; ++i)
        du[i] = limdiff(u[i-1], u[i], u[i+1]);
}


// Compute limited derivs across stride
static inline
void __attribute__((target(mic))) limited_derivk(float* restrict du,
                    const float* restrict u,
                    int ncell, int stride)
{
    assert(stride > 0);
    // #pragma vector aligned
    for (int i = 0; i < ncell; ++i)
        du[i] = limdiff(u[i-stride], u[i], u[i+stride]);
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
 *
 * We're slightly tricky in the corrector in that we write
 * $$
 *   v(i,j) = (s(i+1,j) + s(i,j)) - (d(i+1,j)-d(i,j))
 * $$
 * where $s(i,j)$ comprises the $u$ and $x$-derivative terms in the
 * update formula, and $d(i,j)$ the $y$-derivative terms.  This cuts
 * the arithmetic cost a little (not that it's that big to start).
 * It also makes it more obvious that we only need four rows worth
 * of scratch space.
 */


// Predictor half-step
static
void __attribute__((target(mic))) central2d_predict(float* restrict v,
                       float* restrict scratch,
                       const float* restrict u,
                       const float* restrict f,
                       const float* restrict g,
                       float dtcdx2, float dtcdy2,
                       int nx, int ny, int nfield)
{
    float* restrict fx = scratch;
    float* restrict gy = scratch+nx;
    // #pragma vector aligned
    for (int k = 0; k < nfield; ++k) {
        // #pragma vector aligned
        for (int iy = 1; iy < ny-1; ++iy) {
            int offset = (k*ny+iy)*nx+1;
            limited_deriv1(fx+1, f+offset, nx-2);
            limited_derivk(gy+1, g+offset, nx-2, nx);
            // #pragma vector aligned
            for (int ix = 1; ix < nx-1; ++ix) {
                int offset = (k*ny+iy)*nx+ix;
                v[offset] = u[offset] - dtcdx2 * fx[ix] - dtcdy2 * gy[ix];
            }
        }
    }
}


// Corrector
static
void __attribute__((target(mic))) central2d_correct_sd(float* restrict s,
                          float* restrict d,
                          const float* restrict ux,
                          const float* restrict uy,
                          const float* restrict u,
                          const float* restrict f,
                          const float* restrict g,
                          float dtcdx2, float dtcdy2,
                          int xlo, int xhi)
{
    // #pragma vector aligned
    for (int ix = xlo; ix < xhi; ++ix)
        s[ix] =
            0.2500f * (u [ix] + u [ix+1]) +
            0.0625f * (ux[ix] - ux[ix+1]) +
            dtcdx2  * (f [ix] - f [ix+1]);
    // #pragma vector aligned
    for (int ix = xlo; ix < xhi; ++ix)
        d[ix] =
            0.0625f * (uy[ix] + uy[ix+1]) +
            dtcdy2  * (g [ix] + g [ix+1]);
}


// Corrector
static
void __attribute__((target(mic))) central2d_correct(float* restrict v,
                       float* restrict scratch,
                       const float* restrict u,
                       const float* restrict f,
                       const float* restrict g,
                       float dtcdx2, float dtcdy2,
                       int xlo, int xhi, int ylo, int yhi,
                       int nx, int ny, int nfield)
{
    assert(0 <= xlo && xlo < xhi && xhi <= nx);
    assert(0 <= ylo && ylo < yhi && yhi <= ny);

    float* restrict ux = scratch;
    float* restrict uy = scratch +   nx;
    float* restrict s0 = scratch + 2*nx;
    float* restrict d0 = scratch + 3*nx;
    float* restrict s1 = scratch + 4*nx;
    float* restrict d1 = scratch + 5*nx;
    // #pragma vector aligned
    for (int k = 0; k < nfield; ++k) {

        float*       restrict vk = v + k*ny*nx;
        const float* restrict uk = u + k*ny*nx;
        const float* restrict fk = f + k*ny*nx;
        const float* restrict gk = g + k*ny*nx;

        limited_deriv1(ux+1, uk+ylo*nx+1, nx-2);
        limited_derivk(uy+1, uk+ylo*nx+1, nx-2, nx);
        central2d_correct_sd(s1, d1, ux, uy,
                             uk + ylo*nx, fk + ylo*nx, gk + ylo*nx,
                             dtcdx2, dtcdy2, xlo, xhi);
        // #pragma vector aligned
        for (int iy = ylo; iy < yhi; ++iy) {

            float* tmp;
            tmp = s0; s0 = s1; s1 = tmp;
            tmp = d0; d0 = d1; d1 = tmp;

            limited_deriv1(ux+1, uk+(iy+1)*nx+1, nx-2);
            limited_derivk(uy+1, uk+(iy+1)*nx+1, nx-2, nx);
            central2d_correct_sd(s1, d1, ux, uy,
                                 uk + (iy+1)*nx, fk + (iy+1)*nx, gk + (iy+1)*nx,
                                 dtcdx2, dtcdy2, xlo, xhi);
            // #pragma vector aligned
            for (int ix = xlo; ix < xhi; ++ix)
                vk[iy*nx+ix] = (s1[ix]+s0[ix])-(d1[ix]-d0[ix]);
        }
    }
}


static
void __attribute__((target(mic))) central2d_step(float* restrict u, float* restrict v,
                    float* restrict scratch,
                    float* restrict f,
                    float* restrict g,
                    int io, int nx, int ny, int ng,
                    int nfield,
                    float dt, float dx, float dy)
{
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;

    float dtcdx2 = 0.5 * dt / dx;
    float dtcdy2 = 0.5 * dt / dy;

    shallow2d_flux(f, g, u, nx_all * ny_all, nx_all * ny_all);

    central2d_predict(v, scratch, u, f, g, dtcdx2, dtcdy2,
                      nx_all, ny_all, nfield);

    // Flux values of f and g at half step
    // #pragma vector aligned
    for (int iy = 1; iy < ny_all-1; ++iy) {
        int jj = iy*nx_all+1;
        shallow2d_flux(f+jj, g+jj, v+jj, nx_all-2, nx_all * ny_all);
    }

    central2d_correct(v+io*(nx_all+1), scratch, u, f, g, dtcdx2, dtcdy2,
                      1, nx_all,
                      1, ny_all,
                      nx_all, ny_all, nfield);

    // Copy from v storage back to main grid
    memcpy(u,
           v,
           (nfield*ny_all) * nx_all * sizeof(float));
}


// p is processor number, which determines block location in global memory
// p is row major, e.g.
// 1 2
// 3 4
void __attribute__((target(mic))) copy_to_block(int offset_x, int offset_y, central2d_t* block, float* u, int nx, int ny, int ng, int nfield) {
    int nx_all_block = (block->ng*2 + block->nx);
    int ny_all_block = block->ng*2 + block->ny;
    // #pragma vector aligned
    for (int k = 0; k < nfield; ++k) {
        // #pragma vector aligned
        for (int iy = 0; iy < ny_all_block; ++iy) {
            // #pragma vector aligned
            for(int ix = 0; ix < nx_all_block; ++ix) {
                block->u[central2d_offset_absolute(block->nx, block->ny, block->ng, k, ix, iy)] =
                            u[central2d_offset_absolute(nx, ny, ng, k, ix+offset_x, iy+offset_y)];
            }
        }
    }
}

void __attribute__((target(mic))) copy_to_global(int offset_x, int offset_y, central2d_t* block, float* u, int nx, int ny, int ng, int nfield) {
    int nx_block = block->nx;
    int ny_block = block->ny;

    // #pragma vector aligned
    for (int k = 0; k < nfield; ++k) {
        // #pragma vector aligned
        for (int iy = 0; iy < ny_block; ++iy) {
            // #pragma vector aligned
            for(int ix = 0; ix < nx_block; ++ix) {
                u[central2d_offset(nx, ny, ng, k, ix+offset_x, iy+offset_y)] =
                        block->u[central2d_offset(block->nx, block->ny, block->ng, k, ix, iy)];
            }
        }
    }
}

void __attribute__((target(mic))) offsets(int* offset_x, int* offset_y, int p_number, int p_total, int nx, int ny) {
    int p_dim = (int) sqrt(p_total);
    int block_x = nx / p_dim;
    int block_y = ny / p_dim;

    int p_x = p_number % p_dim;
    int p_y = p_number / p_dim;

    *offset_x = p_x * block_x;
    *offset_y = p_y * block_y;
}

// void __attribute__((target(mic))) print_block(central2d_t* block) {
//     int nx_all_block = (block->ng*2 + block->nx);
//     int ny_all_block = block->ng*2 + block->ny;
//     for (int iy = 0; iy < ny_all_block; ++iy) {
//         for(int ix = 0; ix < nx_all_block; ++ix) {
//             printf("%0.01f ", block->u[central2d_offset_absolute(block, 0, ix, iy)]);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }

/**
 * ### Advance a fixed time
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

int __attribute__((target(mic))) central2d_xrun(float* restrict u,
                   int nx, int ny, int ng,
                   int nfield,
                   float tfinal, float dx, float dy, float cfl,
                   int p, int b) {
    int nstep = 0;
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    int side = (int)sqrt(p);
    int block_nx_all = nx/side + 2*ng;
    int block_ny_all = ny/side + 2*ng;
    bool done = false;
    float t = 0;
    float small_number = 1.0e-15f;
    float global_cxy[2] = {small_number, small_number};
    int rounds = b;
    float dt;
    central2d_periodic(u,nx,ny,ng,nfield);
    omp_set_dynamic(0);
    omp_set_num_threads(p);
    #pragma omp parallel
    {
        central2d_t* block = (central2d_t*) malloc(sizeof(central2d_t*));
        block = central2d_init(nx*dx/side, ny*dy/side,
                   nx/side, ny/side, nfield,
                   cfl, b);
        int proc = omp_get_thread_num();
        int offset_x;
        int offset_y;
        float local_cxy[2];
        offsets(&offset_x, &offset_y, proc, p, nx, ny);


        while (!done) {
            // Apply periodic boundary conditions
            #pragma omp single
            {
                central2d_periodic(u,nx,ny,ng,nfield);
            }

            // Copy memory to each subdomain
            copy_to_block(offset_x, offset_y, block, u, nx, ny, ng, nfield);

            // Calculate speed for each subdomain
            local_cxy[0] = small_number;
            local_cxy[1] = small_number;
            shallow2d_speed(local_cxy, block->u, block_nx_all*block_ny_all, block_nx_all*block_ny_all);

            // Find maximum speed over all subdomains
            #pragma omp critical
            {
                global_cxy[0] = fmaxf(global_cxy[0], local_cxy[0]);
                global_cxy[1] = fmaxf(global_cxy[1], local_cxy[1]);
            }

            // We need to make sure global_cxy is accurate before calculating dt
            #pragma omp barrier

            // Use maximum speed to calculate largest time step we can safely take
            #pragma omp single
            {
                dt = cfl / fmaxf(global_cxy[0]/dx, global_cxy[1]/dy);
                if(rounds > 1) {
                    dt = 0.9*dt; // TODO: set this more intelligently?
                }
                if (t + 2*rounds*dt >= tfinal) {
                    dt = (tfinal-t)/(2*rounds);
                    done = true;
                }

                // Feels a little like cheating to update these before actually computing steps, but I think it works
                t += 2*rounds*dt;
                nstep += 2*rounds;

                // Also need to reset global_cxy somewhere where we have only one thread
                global_cxy[0] = small_number;
                global_cxy[1] = small_number;
            }

            // run each subdomain for b*2 steps
            for(int i = 0; i < rounds; i++) {
                central2d_step(block->u, block->v, block->scratch, block->f, block->g,
                               0, block->nx, block->ny, block->ng,
                               block->nfield,
                               dt, block->dx, block->dy);
                central2d_step(block->u, block->v, block->scratch, block->f, block->g,
                               1, block->nx, block->ny, block->ng,
                               block->nfield,
                               dt, block->dx, block->dy);
            }

            // copy memory to global sim
            copy_to_global(offset_x, offset_y, block, u, nx, ny, ng, nfield);
	    
	    #pragma omp barrier
	    
            // Update time and number of steps
            // #pragma omp single
            // {
            //     t += 2*rounds*dt;
            //     nstep += 2*rounds;
            // }
        }
        central2d_free(block);
    }
    return nstep;
}

int central2d_run(float* restrict u,
                   int nx, int ny, int ng,
                   int nfield,
                   float tfinal, float dx, float dy, float cfl,
                   int p,
                   int b,
                   int mic
                )
{
    int nstep = 0;
    if (mic) {
        int nx_all = nx + 2*ng;
        int ny_all = ny + 2*ng;
        #pragma offload target(mic) inout(u : length(nx_all*ny_all*nfield)) \
        inout(nstep) \
        in(nx, ny, ng, nfield, tfinal, dx, dy, cfl, p, b)
        {
            nstep = central2d_xrun(u, nx, ny, ng, nfield, tfinal, dx, dy, cfl, p, b);
        }
    }
    else {
        nstep = central2d_xrun(u, nx, ny, ng, nfield, tfinal, dx, dy, cfl, p, b);
    }

    return nstep;
}
