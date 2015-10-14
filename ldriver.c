#include "stepper.h"
#include "shallow2d.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

#include <assert.h>
#include <stdio.h>

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
 * ## MinMod limiter
 *
 * Numerical methods for solving nonlinear wave equations are
 * complicated by the fact that even with smooth initial data, a
 * nonlinear wave can develop discontinuities (shocks) in finite time.
 *
 * This makes for interesting analysis, since a "strong" solution
 * that satisfies the differential equation no longer makes sense at
 * a shock -- instead, we have to come up with some mathematically
 * and physically reasonable definition of a "weak" solution that
 * satisfies the PDE away from the shock and satisfies some other
 * condition (an entropy condition) at the shock.
 *
 * The presence of shocks also makes for interesting *numerical*
 * analysis, because we need to be careful about employing numerical
 * differentiation formulas that sample a discontinuous function at
 * points on different sides of a shock.  Using such formulas naively
 * usually causes the numerical method to become unstable.  A better
 * method -- better even in the absence of shocks! -- is to consider
 * multiple numerical differentiation formulas and use the highest
 * order one that "looks reasonable" in the sense that it doesn't
 * predict wildly larger slopes than the others.  Because these
 * combined formulas *limit* the wild behavior of derivative estimates
 * across a shock, we call them *limiters*.  With an appropriate limiter,
 * we can construct methods that have high-order accuracy away from shocks
 * and are at least first-order accurate close to a shock.  These are
 * sometimes called *high-resolution* methods.
 *
 * The MinMod (minimum modulus) limiter is one example of a limiter.
 * The MinMod limiter estimates the slope through points $f_-, f_0, f_+$
 * (with the step $h$ scaled to 1) by
 * $$
 *   f' = \operatorname{minmod}((f_+-f_-)/2, \theta(f_+-f_0), \theta(f_0-f_-))
 * $$
 * where the minmod function returns the argument with smallest absolute
 * value if all arguments have the same sign, and zero otherwise.
 * Common choices of $\theta$ are $\theta = 1.0$ and $\theta = 2.0$.
 *
 * The minmod limiter *looks* like it should be expensive to computer,
 * since superficially it seems to require a number of branches.
 * We do something a little tricky, getting rid of the condition
 * on the sign of the arguments using the `copysign` instruction.
 * If the compiler does the "right" thing with `max` and `min`
 * for floating point arguments (translating them to branch-free
 * intrinsic operations), this implementation should be relatively fast.
 *
 * There are many other potential choices of limiters as well.  We'll
 * stick with this one for the code, but you should feel free to
 * experiment with others if you know what you're doing and think it
 * will improve performance or accuracy.
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
 * The `Central2D` solver class takes the template arguments `Physics`
 * that defines:
 *
 *  - A type for numerical data (`real`)
 *  - A type for solution and flux vectors in each cell (`vec`)
 *  - A flux computation function (`flux(vec& F, vec& G, const vec& U)`)
 *  - A wave speed computation function
 *    (`wave_speed(real& cx, real& cy, const vec& U)`).
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

void solution_check(central2d_t* sim)
{
    int nx = sim->nx, ny = sim->ny;
    float* u = sim->u;
    float h_sum = 0, hu_sum = 0, hv_sum = 0;
    float hmin = u[central2d_offset(sim,0,0,0)];
    float hmax = hmin;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            float h = u[central2d_offset(sim,0,i,j)];
            h_sum += h;
            hu_sum += u[central2d_offset(sim,1,i,j)];
            hv_sum += u[central2d_offset(sim,2,i,j)];
            hmax = fmaxf(h, hmax);
            hmin = fminf(h, hmin);
        }
    float cell_area = sim->dx * sim->dy;
    h_sum *= cell_area;
    hu_sum *= cell_area;
    hv_sum *= cell_area;
    printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n",
           h_sum, hu_sum, hv_sum, hmin, hmax);
    assert(hmin > 0);
}

/**
 * ## I/O
 *
 * After finishing a run (or every several steps), we might want to
 * write out a data file for further processing by some other program
 * -- in this case, a Python visualizer.  The visualizer takes the
 * number of pixels in x and y in the first two entries, then raw
 * single-precision raster pictures.
 */

FILE* viz_open(const char* fname, central2d_t* sim)
{
    FILE* fp = fopen(fname, "w");
    if (fp) {
        float xy[2] = {sim->nx, sim->ny};
        fwrite(xy, sizeof(float), 2, fp);
    }
    return fp;
}

void viz_close(FILE* fp)
{
    fclose(fp);
}

void viz_frame(FILE* fp, central2d_t* sim)
{
    if (fp)
        for (int j = 0; j < sim->ny; ++j)
            for (int i = 0; i < sim->nx; ++i) {
                float* uij = sim->u + central2d_offset(sim,0,i,j);
                fwrite(uij, sizeof(float), 1, fp);
            }
}

/**
 * # Lua driver routines
 *
 * A better way to manage simulation parameters is by a scripting
 * language.  Python is a popular choice, but I prefer Lua for many
 * things (not least because it is an easy build).  It's also quite
 * cheap to call a Lua function for every point in a mesh
 * (less so for Python, though it probably won't make much difference).
 *
 * ## Lua helpers
 *
 * We want to be able to get numbers and strings with a default value
 * when nothing is specified.  Lua 5.3 has this as a built-in, I think,
 * but the following codes are taken from earlier versions of Lua.
 */

double lget_number(lua_State* L, const char* name, double x)
{
    lua_getfield(L, 1, name);
    if (lua_type(L, -1) != LUA_TNIL) {
        if (lua_type(L, -1) != LUA_TNUMBER)
            luaL_error(L, "Expected %s to be a number", name);
        x = lua_tonumber(L, -1);
    }
    lua_pop(L, 1);
    return x;
}


int lget_int(lua_State* L, const char* name, int x)
{
    lua_getfield(L, 1, name);
    if (lua_type(L, -1) != LUA_TNIL) {
        if (lua_type(L, -1) != LUA_TNUMBER)
            luaL_error(L, "Expected %s to be a number", name);
        x = lua_tointeger(L, -1);
    }
    lua_pop(L, 1);
    return x;
}


const char* lget_string(lua_State* L, const char* name, const char* x)
{
    lua_getfield(L, 1, name);
    if (lua_type(L, -1) != LUA_TNIL) {
        if (lua_type(L, -1) != LUA_TSTRING)
            luaL_error(L, "Expected %s to be a string", name);
        x = lua_tostring(L, -1);
    }
    lua_pop(L, 1);
    return x;
}


/**
 * ## Lua callback functions
 *
 * We specify the initial conditions by providing the simulator
 * with a callback function to be called at each cell center.
 */

void lua_init_sim(lua_State* L, central2d_t* sim)
{
    lua_getfield(L, 1, "init");
    if (lua_type(L, -1) != LUA_TFUNCTION)
        luaL_error(L, "Expected init to be a string");

    int nx = sim->nx, ny = sim->ny, nfield = sim->nfield;
    float dx = sim->dx, dy = sim->dy;
    float* u = sim->u;

    for (int ix = 0; ix < nx; ++ix) {
        float x = (ix + 0.5) * dx;
        for (int iy = 0; iy < ny; ++iy) {
            float y = (iy + 0.5) * dy;
            lua_pushvalue(L, -1);
            lua_pushnumber(L, x);
            lua_pushnumber(L, y);
            lua_call(L, 2, nfield);
            for (int k = 0; k < nfield; ++k)
                u[central2d_offset(sim,k,ix,iy)] = lua_tonumber(L, k-nfield);
            lua_pop(L, nfield);
        }
    }

    lua_pop(L,1);
}


/**
 * ## Running the simulation
 *
 * The `run_sim` function looks a lot like the main routine of the
 * "ordinary" command line driver.
 * We can specify the initial conditions by providing the simulator
 * with a callback function to be called at each cell center.
 * There's nothing wrong with writing that callback in C++, but we
 * do need to make sure to keep the Lua state as context.  It's not
 * so easy to store a Lua function directly in C++, but we can store
 * it in a special registry table in Lua (where the key is the "this"
 * pointer for the object).
 */

int run_sim(lua_State* L)
{
    int n = lua_gettop(L);
    if (n != 1 || !lua_istable(L, 1))
        luaL_error(L, "Argument must be a table");

    double w = lget_number(L, "w", 2.0);
    double h = lget_number(L, "h", w);
    double cfl = lget_number(L, "cfl", 0.45);
    double ftime = lget_number(L, "ftime", 0.01);
    int nx = lget_int(L, "nx", 200);
    int ny = lget_int(L, "ny", nx);
    int frames = lget_int(L, "frames", 50);
    const char* fname = lget_string(L, "out", "sim.out");

    central2d_t* sim = central2d_init(w,h, nx,ny,
                                      3, shallow2d_flux, shallow2d_speed, cfl);
    lua_init_sim(L,sim);

    printf("%g %g %d %d %g %d %g\n", w, h, nx, ny, cfl, frames, ftime);
    FILE* viz = viz_open(fname, sim);
    solution_check(sim);
    viz_frame(viz, sim);
    for (int i = 0; i < frames; ++i) {
#ifdef _OPENMP
        double t0 = omp_get_wtime();
        central2d_run(ftime, sim);
        double t1 = omp_get_wtime();
        printf("Time: %e\n", t1-t0);
#else
        central2d_run(ftime, sim);
#endif
        solution_check(sim);
        viz_frame(viz, sim);
    }
    central2d_free(sim);
    return 0;
}


int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s fname args\n", argv[0]);
        return -1;
    }

    lua_State* L = luaL_newstate();
    luaL_openlibs(L);
    lua_register(L, "simulate", run_sim);

    lua_newtable(L);
    for (int i = 2; i < argc; ++i) {
        lua_pushstring(L, argv[i]);
        lua_rawseti(L, 1, i-1);
    }
    lua_setglobal(L, "args");

    if (luaL_dofile(L, argv[1]))
        printf("%s\n", lua_tostring(L,-1));
    lua_close(L);
    return 0;
}
