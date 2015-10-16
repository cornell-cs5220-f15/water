#ifndef STEPPER_H
#define STEPPER_H

#include <math.h>

//ldoc
/**
 * ## Interface
 *
 * ### Physics function types
 *
 * From the perspective of the solver, the physics can be characterized
 * by two functions: the flux function and the max wave speed function
 * (used to control the time step).  We define callback types for these
 * two functions, with the assumption that the different components
 * of the solution and fluxes are separated by `field_stride`.
 *
 */
typedef void (*flux_t)(float* FU, float* GU, const float* U,
                       int ncell, int field_stride);
typedef void (*speed_t)(float* cxy, const float* U,
                        int ncell, int field_stride);


/**
 * ### Solver data structure
 *
 * The solver has a number of parameters that define the physics
 * (the `flux` and `speed` functions mentioned above as well as
 * the number of fields `nfield`); the spatial discretization
 * (`nx`, `ny`, `ng`, `dx`, `dy`); and the time discretization (`cfl`).
 * In addition, we have storage for the current solution and for
 * various quantities that are internally important (fluxes and
 * numerical derivatives).  These latter should probably be hidden,
 * but I haven't done so yet.
 *
 */
typedef struct central2d_t {

    int nfield;   // Number of components in system
    int nx, ny;   // Grid resolution in x/y (without ghost cells)
    int ng;       // Number of ghost cells
    float dx, dy; // Cell width in x/y
    float cfl;    // Max allowed CFL number

    // Flux and speed functions
    flux_t flux;
    speed_t speed;

    // Storage
    float* u;
    float* v;
    float* f;
    float* g;
    float* scratch;

} central2d_t;


/**
 * For the most part, we treat the `central2d_t` as a read-only
 * structure.  The exceptions are the constructor and destructor
 * functions.
 */
central2d_t* central2d_init(float w, float h, int nx, int ny,
                            int nfield, flux_t flux, speed_t speed,
                            float cfl);
void central2d_free(central2d_t* sim);

/**
 * For initialization and for reporting on the solution, it's helpful
 * to expose how indexing is done.  We manage this with an offset
 * function.  Here `k` is the field index, and `(ix,iy)` are the
 * (zero-based) cell index, where cell `(0,0)` is a corner
 * real (non-ghost) cell.
 */
int  central2d_offset(central2d_t* sim, int k, int ix, int iy);

/**
 * ### Running the simulation
 *
 * The `central2d_run` function advances the simulation from the
 * current state by time `tfinal`.  It returns the number of steps
 * taken, determined by the CFL restriction and by the requirement
 * that we always take steps in multiples of two so that we end
 * at the reference grid.
 */
int central2d_run(central2d_t* sim, float tfinal);

/**
 * ### Applying boundary conditions
 *
 * Ideally, we would not be applying boundary conditions inside the
 * solver, but through an external call (as we do with the other
 * cases).  This is because, apart from periodic boundary conditions,
 * the way that we manipulate ghost cell data in order to enforce BCs
 * usually depends a bit on the BCs that are appropriate to the physics
 * (which may vary from field to field).  For this exercise, we're always
 * going to use periodic BCs.  But I want to leave the interface function
 * public in the eventuality that I might swap in a function pointer
 * for applying the BCs.
 */
void central2d_periodic(float* u, int nx, int ny, int ng, int nfield);

//ldoc off
#endif /* STEPPER_H */
