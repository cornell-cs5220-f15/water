#include "stepper.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <omp.h>

//ldoc on
/**
 * ## Implementation
 *
 * ### Structure allocation
 */

central2d_t* central2d_init(float w, float h, int nx, int ny,
                            int nfield, flux_t flux, speed_t speed,
                            float cfl)
{
    // This should be an automatic update but I don't want to define global variable, change this in tuning!!!!
    // The number of ghost celss should be 3*2*iter
    // where iter is the number of iteration before sync in central2d_step()
    int ng = 16; // # of ghost cells ( 4*iter for laziness and safety)

    central2d_t* sim = (central2d_t*) malloc(sizeof(central2d_t));
    sim->nx = nx; // dimension size in x
    sim->ny = ny;
    sim->ng = ng; // number of ghost cells
    sim->nfield = nfield; // each vector has three components
    sim->dx = w/nx; // Grid size in x
    sim->dy = h/ny;
    sim->flux = flux; // flux ???
    sim->speed = speed; // speed ???
    sim->cfl = cfl; // CFL prefix coefficient

    int nx_all = nx + 2*ng; // ghost cells on each side to avoid sync
    int ny_all = ny + 2*ng;
    int nc = nx_all * ny_all; // entire space
    int N  = nfield * nc; // how many entries for each vector
    sim->u  = (float*) malloc((4*N + 6*nx_all)* sizeof(float)); // allocate all space,not quite sure what the 6*nx_all are for (scratch? what is it?)
    sim->v  = sim->u +   N; // storage space for half step grid
    sim->f  = sim->u + 2*N;
    sim->g  = sim->u + 3*N;
    sim->scratch = sim->u + 4*N;

    return sim;
}


void central2d_free(central2d_t* sim)
{
    free(sim->u);
    free(sim);
}


int central2d_offset(central2d_t* sim, int k, int ix, int iy)
{
    int nx = sim->nx, ny = sim->ny, ng = sim->ng;
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    return (k*ny_all+(ng+iy))*nx_all+(ng+ix);
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
void copy_subgrid(float* restrict dst,
                  const float* restrict src,
                  int nx, int ny, int stride)
{
    // from src, copy nx * ny date subblock to dst.
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix)
            dst[iy*stride+ix] = src[iy*stride+ix]; // Variable stride is used to accomodate the existence of ghost cells in the data
}

void central2d_periodic(float* restrict u,
                        int nx, int ny, int ng, int nfield)
{
    // Stride and number per field
    int s = nx + 2*ng; // the dimension size in x when ghost cells are present
    int field_stride = (ny+2*ng)*s; // the step size in y to cross the one subdomain

    // Offsets of left, right, top, and bottom data blocks and ghost blocks
    int l = nx,   lg = 0;
    int r = ng,   rg = nx+ng;
    int b = ny*s, bg = 0;
    int t = ng*s, tg = (nx+ng)*s; // should it be tg = (ny+ng)*s ? It doesn't matter for now because nx = ny

    // Copy data into ghost cells on each side
    for (int k = 0; k < nfield; ++k) {
        float* uk = u + k*field_stride;
        copy_subgrid(uk+lg, uk+l, ng, ny+2*ng, s); // for periodic condition update
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
float xmin2s(float s, float a, float b) {
    float sa = copysignf(s, a);
    float sb = copysignf(s, b);
    float abs_a = fabsf(a);
    float abs_b = fabsf(b);
    float min_abs = (abs_a < abs_b ? abs_a : abs_b);
    return (sa+sb) * min_abs;
}


// Limited combined slope estimate
static inline
float limdiff(float um, float u0, float up) {
    const float theta = 2.0;
    const float quarter = 0.25;
    float du1 = u0-um;   // Difference to left
    float du2 = up-u0;   // Difference to right
    float duc = up-um;   // Twice centered difference
    return xmin2s( quarter, xmin2s(theta, du1, du2), duc );
}


// Compute limited derivs
static inline
void limited_deriv1(float* restrict du,
                    const float* restrict u,
                    int ncell)
{
    for (int i = 0; i < ncell; ++i)
        du[i] = limdiff(u[i-1], u[i], u[i+1]);
}


// Compute limited derivs across stride
static inline
void limited_derivk(float* restrict du,
                    const float* restrict u,
                    int ncell, int stride)
{
    assert(stride > 0);
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
void central2d_predict(float* restrict v,
                       float* restrict scratch,
                       const float* restrict u,
                       const float* restrict f,
                       const float* restrict g,
                       float dtcdx2, float dtcdy2,
                       int nx, int ny, int nfield)
{
    float* restrict fx = scratch;
    float* restrict gy = scratch+nx;
    for (int k = 0; k < nfield; ++k) {
        for (int iy = 1; iy < ny-1; ++iy) {
            int offset = (k*ny+iy)*nx+1;
            limited_deriv1(fx+1, f+offset, nx-2);
            limited_derivk(gy+1, g+offset, nx-2, nx);
            for (int ix = 1; ix < nx-1; ++ix) {
                int offset = (k*ny+iy)*nx+ix;
                v[offset] = u[offset] - dtcdx2 * fx[ix] - dtcdy2 * gy[ix];
            }
        }
    }
}


// Corrector
static
void central2d_correct_sd(float* restrict s,
                          float* restrict d,
                          const float* restrict ux,
                          const float* restrict uy,
                          const float* restrict u,
                          const float* restrict f,
                          const float* restrict g,
                          float dtcdx2, float dtcdy2,
                          int xlo, int xhi)
{
    for (int ix = xlo; ix < xhi; ++ix)
        s[ix] =
            0.2500f * (u [ix] + u [ix+1]) +
            0.0625f * (ux[ix] - ux[ix+1]) +
            dtcdx2  * (f [ix] - f [ix+1]);
    for (int ix = xlo; ix < xhi; ++ix)
        d[ix] =
            0.0625f * (uy[ix] + uy[ix+1]) +
            dtcdy2  * (g [ix] + g [ix+1]);
}


// Corrector
static
void central2d_correct(float* restrict v,
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

        for (int iy = ylo; iy < yhi; ++iy) {

            float* tmp;
            tmp = s0; s0 = s1; s1 = tmp;
            tmp = d0; d0 = d1; d1 = tmp;

            limited_deriv1(ux+1, uk+(iy+1)*nx+1, nx-2);
            limited_derivk(uy+1, uk+(iy+1)*nx+1, nx-2, nx);
            central2d_correct_sd(s1, d1, ux, uy,
                                 uk + (iy+1)*nx, fk + (iy+1)*nx, gk + (iy+1)*nx,
                                 dtcdx2, dtcdy2, xlo, xhi);

            for (int ix = xlo; ix < xhi; ++ix)
                vk[iy*nx+ix] = (s1[ix]+s0[ix])-(d1[ix]-d0[ix]);
        }
    }
}


static
void central2d_step(float* restrict u, float* restrict v,
                    float* restrict scratch,
                    float* restrict f,
                    float* restrict g,
                    int io, int nx, int ny, int ng,
                    int nfield, flux_t flux, speed_t speed,
                    float dt, float dx, float dy)
{
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;

    float dtcdx2 = 0.5 * dt / dx;
    float dtcdy2 = 0.5 * dt / dy;

    flux(f, g, u, nx_all * ny_all, nx_all * ny_all);

    central2d_predict(v, scratch, u, f, g, dtcdx2, dtcdy2,
                      nx_all, ny_all, nfield);

    // Flux values of f and g at half step
    for (int iy = 1; iy < ny_all-1; ++iy) {
        int jj = iy*nx_all+1;
        flux(f+jj, g+jj, v+jj, nx_all-2, nx_all * ny_all);
    }
    //flux(f, g, v, nx_all * ny_all, nx_all * ny_all);

    central2d_correct(v + io*(nx_all+1), scratch, u, f, g, dtcdx2, dtcdy2,
                      //1-io, nx+2*ng-io,
                      //1-io, ny+2*ng-io,
                      ng-io, nx+ng-io,
                      ng-io, ny+ng-io,
                      nx_all, ny_all, nfield);

    // Copy from v storage back to main grid
    //memcpy(u, v, nfield*ny_all*nx_all*sizeof(float)); // copy everything instead, do not update selectively
    //memcpy(u+(ng   )*nx_all+ng,
           //v+(ng-io)*nx_all+ng-io,
           //(nfield*ny_all-ng) * nx_all * sizeof(float));
}

// My subdomain copy function
void copy_subdomain(float ** u_s, float ** v_s,
                float ** f_s, float ** g_s, float ** scratch_s,
                int *ny_sub, float* restrict u, int nx, int ny, int ng,
                int nfield, int index, int ndomain){
        int new_ny; // The size of subdomain
        int subsize = ceil( ny / ndomain);
        if(index < ndomain-1) new_ny = subsize;
        else new_ny = ny - (ndomain-1)*subsize;
        (*ny_sub) = new_ny;

        int nx_all = nx + 2*ng;
        int ny_all = new_ny + 2*ng;
        int nc = nx_all * ny_all;
        int N = nfield * nc;
        int start_index = index*subsize*nx_all;
        int Nc = nx_all * (ny + 2*ng);
        // Maybe use _mm_malloc for aligned memory block, suggestion for later
        float* u_new = (float*) malloc((4*N + 6*nx_all)*sizeof(float));
        *u_s = (float*)u_new;
        *v_s = (float*)u_new + N;
        *f_s = (float*)u_new + 2*N;
        *g_s = (float*)u_new + 3*N;
        *scratch_s = (float*)u_new + 4*N;

        for (int k = 0; k < nfield; k++){
                memcpy(*u_s + nc*k, u + start_index + Nc*k, nc*sizeof(float));
                //memcpy(*v_s + nc*k, u + start_index + Nc*k, nc*sizeof(float));
                //memcpy(*f_s + nc*k, u + start_index + Nc*k, nc*sizeof(float));
                //memcpy(*g_s + nc*k, u + start_index + Nc*k, nc*sizeof(float));
        }
}

void sync_subdomain(float* restrict u_s, float* restrict u,
                int ny_sub, int nx, int ny, int ng,
                int nfield, int index, int ndomain){
        int nx_all = nx + 2*ng;
        int ny_all_sub = ny_sub + 2*ng;
        int nc = nx_all * ny_all_sub;
        int Nc = nx_all * (ny + 2*ng);
        int subsize = ceil(ny / ndomain); 
        int start_index = (index*subsize + ng)*nx_all;
        int sub_start = ng*nx_all;
        //printf("For processor %d/%d, the values are ny_sub = %d, start_index = %d, sub_start = %d \n", index, ndomain, ny_sub, start_index, sub_start);
        for (int k = 0; k < nfield ; k++){
                memcpy(u+start_index+Nc*k, u_s+sub_start+nc*k, nx_all*ny_sub*sizeof(float));
        } // Copy the real data part back to the main grid.
}
void update_subdomain(float* restrict u_s, float* restrict u,
                int ny_sub, int nx, int ny, int ng,
                int nfield, int index, int ndomain){
        int s = nx + 2*ng;
        int field_stride = (ny+2*ng)*s;
        int sub_field_stride = (ny_sub+2*ng)*s;
        
        int l = 0, lg = 0;
        int r = nx+ng, rg = nx+ng;
        int b = (ny_sub+ng)*s, bg = (ny_sub+ng)*s;
        int t = 0, tg = 0; // I think this is correct?

        int subsize = ceil(ny / ndomain); 
        int start_index = (index*subsize)*s; // no ng because of the other index system we have
        //printf("For processor %d/%d, the values are ny_sub = %d, start_index = %d\n", index, ndomain, ny_sub, start_index);
        for (int k = 0; k < nfield; k++){
                float* uk = u + k*field_stride + start_index;
                float* u_sk = u_s + k*sub_field_stride;
                copy_subgrid(u_sk+lg, uk+l, ng, ny_sub + 2*ng, s);
                copy_subgrid(u_sk+rg, uk+r, ng, ny_sub + 2*ng, s);
                copy_subgrid(u_sk+tg, uk+t, nx + 2*ng, ng, s);
                copy_subgrid(u_sk+bg, uk+b, nx + 2*ng, ng, s);
        }
}
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

static
int central2d_xrun(float* restrict u, float* restrict v,
                   float* restrict scratch,
                   float* restrict f,
                   float* restrict g,
                   int nx, int ny, int ng,
                   int nfield, flux_t flux, speed_t speed,
                   float tfinal, float dx, float dy, float cfl)
{
    // OMP session should start here
    // sub-domain parallel

    int nstep = 0;
    int nx_all = nx + 2*ng;
    int ny_all = ny + 2*ng;
    bool done = false;
    float t = 0;

    // Initialize the new subdomain vectors here.
#ifdef _OPENMP
    int num_threads_used = 4;
    omp_set_num_threads(num_threads_used);

    float* u_sub[num_threads_used];
    float* v_sub[num_threads_used];
    float* f_sub[num_threads_used];
    float* g_sub[num_threads_used];
    float* scratch_sub[num_threads_used];
    int ny_sub[num_threads_used];
    for (int index = 0; index < num_threads_used; index++){
            copy_subdomain( &u_sub[index], &v_sub[index], &f_sub[index], &g_sub[index], &scratch_sub[index], &ny_sub[index], u, nx, ny, ng, nfield, index, num_threads_used);
    }
    float dt;
#pragma omp parallel shared(dt,dt)
    while (!done) {
//#pragma omp single
        float cxy[2] = {1.0e-15f, 1.0e-15f};
        central2d_periodic(u, nx, ny, ng, nfield); 
        speed(cxy, u, nx_all * ny_all, nx_all * ny_all); 
        //float dt = cfl / fmaxf(cxy[0]/dx, cxy[1]/dy);
        dt = cfl / fmaxf(cxy[0]/dx, cxy[1]/dy);
        // For loops
        int it;
        int iter = 4;
        int idx = omp_get_thread_num();
//#pragma omp barrier
        update_subdomain(u_sub[idx], u, ny_sub[idx], nx, ny, ng, nfield, idx, num_threads_used);
        for(it = 0; it < iter; it ++){
            if (t + 2*dt >= tfinal) {
                dt = (tfinal-t)/2;
                done = true;
            }
            //central2d_step(u_sub[idx], v_sub[idx], scratch_sub[idx],
            //               f_sub[idx], g_sub[idx],
            //               0, nx, ny_sub[idx], ng,
            //               nfield, flux, speed,
            //               dt, dx, dy);
            //central2d_step(u_sub[idx], v_sub[idx], scratch_sub[idx],
            //               f_sub[idx], g_sub[idx],
            //               1, nx, ny_sub[idx], ng,
            //               nfield, flux, speed,
            //               dt, dx, dy);
            //central2d_step(u_sub[idx], v_sub[idx], scratch_sub[idx],
            //               f_sub[idx], g_sub[idx],
            //               0, nx, ny_sub[idx], ng,
            //               nfield, flux, speed,
            //               dt, dx, dy);
            //central2d_step(v_sub[idx], u_sub[idx], scratch_sub[idx],
            //               f_sub[idx], g_sub[idx],
            //               1, nx, ny_sub[idx], ng,
            //               nfield, flux, speed,
            //               dt, dx, dy);
            int ng_eff = 4 * (iter-1-it);
            central2d_step(u_sub[idx], v_sub[idx], scratch_sub[idx],
                           f_sub[idx], g_sub[idx],
                           0, nx+4+2*ng_eff, ny_sub[idx]+4+2*ng_eff, ng-2-ng_eff,
                           nfield, flux, speed,
                           dt, dx, dy);
            central2d_step(v_sub[idx], u_sub[idx], scratch_sub[idx],
                           f_sub[idx], g_sub[idx],
                           1, nx+2*ng_eff, ny_sub[idx]+2*ng_eff, ng-ng_eff,
                           nfield, flux, speed,
                           dt, dx, dy);
#pragma omp single
            t += 2*dt;
            nstep += 2;
        }
        sync_subdomain(u_sub[idx], u, ny_sub[idx], nx, ny, ng, nfield, idx, num_threads_used);
#pragma omp barrier
    }
    // Free the subdomain vectors.
    for (int index = 0; index < num_threads_used; index++){
            free(u_sub[index]);
    }
#else
    while (!done) {
        float cxy[2] = {1.0e-15f, 1.0e-15f};
        central2d_periodic(u, nx, ny, ng, nfield); // Apply periodic boundary condition
        speed(cxy, u, nx_all * ny_all, nx_all * ny_all); 
        float dt = cfl / fmaxf(cxy[0]/dx, cxy[1]/dy);

        // For loops
        int it;
        int iter = 3;
        for(it = 0; it < iter; it ++){
            if (t + 2*dt >= tfinal) {
                dt = (tfinal-t)/2;
                done = true;
            }
            //central2d_step(u, v, scratch, f, g,
            //               0, nx, ny, ng,
            //               nfield, flux, speed,
            //               dt, dx, dy);
            ////central2d_periodic(u, nx, ny, ng, nfield);
            //central2d_step(u, v, scratch, f, g,
            //               1, nx, ny, ng,
            //               nfield, flux, speed,
            //               dt, dx, dy);
            int ng_eff = 4 * (iter-1-it);
            central2d_step(u_sub[idx], v_sub[idx], scratch_sub[idx],
                           f_sub[idx], g_sub[idx],
                           0, nx+4+2*ng_eff, ny_sub[idx]+4+2*ng_eff, ng-2-ng_eff,
                           nfield, flux, speed,
                           dt, dx, dy);
            central2d_step(v_sub[idx], u_sub[idx], scratch_sub[idx],
                           f_sub[idx], g_sub[idx],
                           1, nx+2*ng_eff, ny_sub[idx]+2*ng_eff, ng-ng_eff,
                           nfield, flux, speed,
                           dt, dx, dy);
            t += 2*dt;
            nstep += 2;
        }
    }
#endif
    return nstep;
}


int central2d_run(central2d_t* sim, float tfinal)
{
    return central2d_xrun(sim->u, sim->v, sim->scratch,
                          sim->f, sim->g,
                          sim->nx, sim->ny, sim->ng,
                          sim->nfield, sim->flux, sim->speed,
                          tfinal, sim->dx, sim->dy, sim->cfl);
}

// My own function (not in use)
//central2d_t* central2d_copy_subdomain(central2d_t* sim, int index, int ndomain)
//{
//        int new_ny; // The size of subdomain.
//        int subsize = ceil(sim->ny / ndomain);
//        if(index < ndomain - 1) new_ny = subsize;
//        else new_ny = sim->ny - (ndomain - 1) * subsize;
//
//        central2d_t* copied_sim = (central2d_t*) malloc(sizeof(central2d_t));
//        copied_sim->nx = sim->nx; // nx should be the same
//        copied_sim->ny = new_ny;
//        copied_sim->ng = sim->ng; // Number of ghost cells must be the same for boundary condition
//        copied_sim->nfield = sim->nfield; 
//        copied_sim->dx = sim->dx; // Grid size in x
//        copied_sim->dy = sim->dy;
//        copied_sim->flux = sim->flux; 
//        copied_sim->speed = sim->speed; 
//        copied_sim->cfl = sim->cfl; // CFL prefix coefficient
//
//        int nx_all = copied_sim->nx + 2*copied_sim->ng; // ghost cells on each side to avoid sync
//        int ny_all = copied_sim->ny + 2*copied_sim->ng;
//        int nc = nx_all * ny_all; // entire domain subspace including ghost cells on each sides
//        int N  = copied_sim->nfield * nc; // how many entries for each vector
//        copied_sim->u  = (float*) malloc((4*N + 6*nx_all)* sizeof(float)); // new space for u
//        copied_sim->v  = copied_sim->u + N;
//        copied_sim->f  = copied_sim->u + 2*N;
//        copied_sim->g  = copied_sim->u + 3*N;
//        copied_sim->scratch = copied_sim->u + 4*N;
//        // Stride and number per field
//        int start_index = index*subsize*nx_all; // Starting from the index subdomain with the first ghost cell block considered (the ghost cell will be included)
//        int Nc = nx_all * (sim->ny + 2*sim->ng);
//        for (int k = 0; k < copied_sim->nfield; k++){
//            memcpy(copied_sim->u + nc*k, sim->u+start_index + Nc*k, nc);
//            memcpy(copied_sim->v + nc*k, sim->v+start_index + Nc*k, nc);
//            memcpy(copied_sim->f + nc*k, sim->f+start_index + Nc*k, nc);
//            memcpy(copied_sim->g + nc*k, sim->g+start_index + Nc*k, nc);
//        } // Copy all the memory for subdomain
//        return copied_sim;
//}
//
//void sync_subdomain(float* restrict u, float* restrict u_sub, int nx, int Ny, int ny, int ng, int nfield, int index, int ndomain){
//        int nx_all = nx + 2*ng;
//        int ny_all = ny + 2*ng;
//        int nc = nx_all * ny_all;
//        int Nc = nx_all * (Ny + 2*ng);
//        int subsize = ceil(nx / ndomain); // This should be ny, but for this special case, nx=ny
//        int start_index = (index*subsize + ng)*nx_all;
//        int sub_start = ng*nx;
//        for (int k = 0; k < nfield; k++){
//                memcpy(u + start_index + Nc*k, u_sub + sub_start + nc*k, nx_all*ny);
//        } // Copy the real data part back to the main grid.
//}
