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

typedef Central2D< Shallow2D, MinMod<Shallow2D::real> > Sim;


double lget_number(lua_State* L, const char* name, double x)
{
    lua_getfield(L, 1, name);
    if (lua_type(L, -1) != LUA_TNIL) {
        if (lua_type(L, -1) != LUA_TNUMBER)
            luaL_error(L, "Expected %s to be a number", name);
        double x = lua_tonumber(L, -1);
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
        double x = lua_tointeger(L, -1);
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
        double x = lua_tointeger(L, -1);
    }
    lua_pop(L, 1);
    return x;
}


class SimInitF {
public:

    // Take a function pointer off the top of the stack and save with this
    SimInitF(lua_State* L) : L(L) {
        key = this;
        lua_pushlightuserdata(L, (void*) key);
        lua_pushvalue(L, -2);
        lua_settable(L, LUA_REGISTRYINDEX);
        lua_pop(L, 1);
    }

    void operator()(Sim::vec& f, Sim::real x, Sim::real y) {
        lua_pushlightuserdata(L, (void*) key);
        lua_gettable(L, LUA_REGISTRYINDEX);
        assert(lua_isfunction(L, -1));
        lua_pushnumber(L, x);
        lua_pushnumber(L, y);
        lua_call(L, 2, 1);
        for (int i = 1; i <= f.size(); ++i) {
            lua_rawgeti(L, -1, i);
            f[i-1] = lua_tonumber(L, -1);
            lua_pop(L, 1);
        }
    }

private:
    SimInitF* key;
    lua_State* L;
};


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

    lua_getfield(L, 1, "init");
    if (lua_type(L, -1) != LUA_TFUNCTION)
        luaL_error(L, "Expected init to be a string");        
    SimInitF icfun(L);
    
    Sim sim(w,h, nx,ny, cfl);

    printf("%g %g %d %d %g %d %g\n", w, h, nx, ny, cfl, frames, ftime);
    SimViz<Sim> viz(fname, sim);
    sim.init(icfun);
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
