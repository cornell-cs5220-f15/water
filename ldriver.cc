#include "central2d.h"
#include "shallow2d.h"
#include "minmod.h"
#include "meshio.h"

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

#include <cassert>

//ldoc on
/**
 * # Lua driver routines
 *  
 * A better way to manage simulation parameters is by a scripting
 * language.  Python is a popular choice, but I prefer Lua for many
 * things (not least because it is an easy build).  It's also quite
 * cheap to call a Lua function for every point in a mesh
 * (less so for Python, though it probably won't make much difference).
 * 
 * For the driver, we need to put everything together: we're running
 * a `Central2D` solver for the `Shallow2D` physics with a `MinMod`
 * limiter:
 */

typedef Central2D< Shallow2D, MinMod<Shallow2D::real> > Sim;


/**
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

void lua_init_sim(lua_State* L, Sim& sim)
{
    lua_getfield(L, 1, "init");
    if (lua_type(L, -1) != LUA_TFUNCTION)
        luaL_error(L, "Expected init to be a string");

    for (int ix = 0; ix < sim.xsize(); ++ix) {
        float x = (ix + 0.5) * sim.get_dx();
        for (int iy = 0; iy < sim.ysize(); ++iy) {
            float y = (iy + 0.5) * sim.get_dy();
            lua_pushvalue(L, -1);
            lua_pushnumber(L, x);
            lua_pushnumber(L, y);
            lua_call(L, 2, sim(ix,iy).size());
            for (int k = 0; k < sim(ix,iy).size(); ++k)
                sim(ix,iy)[k] = lua_tonumber(L, k-sim(ix,iy).size());
            lua_pop(L, sim(ix,iy).size());
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

    Sim sim(w,h, nx,ny, cfl);
    lua_init_sim(L,sim);

    printf("%g %g %d %d %g %d %g\n", w, h, nx, ny, cfl, frames, ftime);
    SimViz<Sim> viz(fname, sim);
    sim.solution_check();
    viz.write_frame();
    for (int i = 0; i < frames; ++i) {
#ifdef _OPENMP
        double t0 = omp_get_wtime();
        sim.run(ftime);
        double t1 = omp_get_wtime();
        printf("Time: %e\n", t1-t0);
#else
        sim.run(ftime);
#endif
        sim.solution_check();
        viz.write_frame();
    }
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
