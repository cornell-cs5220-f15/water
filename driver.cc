#include "central2d.h"
#include "shallow2d.h"
#include "minmod.h"
#include "meshio.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <string>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <fstream>

//ldoc on
/**
 * # Driver routines
 *  
 * We use a fairly simple command-line driver to launch this simulation.
 * A better way to do this is to use a scripting language to set up the
 * simulation; Python is a popular choice, though I prefer Lua for many
 * things (not least because it is an easy build).  I may add that
 * capability later; for the moment, it's useful to have a simple
 * command-line interface that ought to run most anywhere.
 * 
 * For the driver, we need to put everything together: we're running
 * a `Central2D` solver for the `Shallow2D` physics with a `MinMod`
 * limiter:
 */

typedef Central2D< Shallow2D, MinMod<Shallow2D::real> > Sim;

/**
 * ## Initial states
 * 
 * Our default problem is a circular dam break problem; the other
 * interesting problem is the wave problem (a wave on a constant
 * flow, starting off smooth and developing a shock in finite time).
 * The pond and river examples should do nothing interesting at all
 * if the numerical method is coded right.
 */

// Circular dam break problem
void dam_break(Sim::vec& u, double x, double y)
{
    x -= 1;
    y -= 1;
    u[0] = 1.0 + 0.5*(x*x + y*y < 0.25+1e-5);
    u[1] = 0;
    u[2] = 0;
}

// Circular dam break problem
// The 3-vector is decoupled for efficiency
void dam_break_h(Sim::real& h, double x, double y)
{
    x -= 1;
    y -= 1;
    h = 1.0 + 0.5*(x*x + y*y < 0.25+1e-5);
}

void dam_break_hu(Sim::real& hu, double x, double y)
{
    hu = 0;
}

void dam_break_hv(Sim::real& hv, double x, double y)
{
    hv = 0;
}

// Still pond (ideally, nothing should move here!)
void pond(Sim::vec& u, double x, double y)
{
    u[0] = 1.0;
    u[1] = 0;
    u[2] = 0;
}

void pond_h(Sim::real& h, double x, double y) {
    h = 1.0;
}

void pond_hu(Sim::real& hu, double x, double y) {
    hu = 0.0;
}

void pond_hv(Sim::real& hv, double x, double y) {
    hv = 0.0;
}

// River (ideally, the solver shouldn't do much with this, either)
void river(Sim::vec& u, double x, double y)
{
    u[0] = 1.0;
    u[1] = 1.0;
    u[2] = 0;
}

void river_h(Sim::real& h, double x, double y) {
    h = 1.0;
}

void river_hu(Sim::real& hu, double x, double y) {
    hu = 1.0;
}

void river_hv(Sim::real& hv, double x, double y) {
    hv = 0.0;
}


// Wave on a river -- develops a shock in finite time!
void wave(Sim::vec& u, double x, double y)
{
    using namespace std;
    u[0] = 1.0 + 0.2 * sin(M_PI*x);
    u[1] = 1.0;
    u[2] = 0;
}

void wave_h(Sim::real& h, double x, double y) {
    h = 1.0 + 0.2 * sin(M_PI*x);
}

void wave_hu(Sim::real& hu, double x, double y) {
    hu = 1.0;
}

void wave_hv(Sim::real& hv, double x, double y) {
    hv = 0.0;
}

/**
 * ## Main driver
 * 
 * Our main driver uses the `getopt` library to parse options,
 * then runs a simulation, writing results to an output file
 * for postprocessing.
 */

int main(int argc, char** argv)
{
    std::string fname = "waves.out";
    std::string ic    = "dam_break";
    int    nx         = 200;
    double width      = 2.0;
    double ftime      = 0.01;
    int    frames     = 50;
    
    int c;
    extern char* optarg;
    while ((c = getopt(argc, argv, "hi:o:n:w:F:f:")) != -1) {
        switch (c) {
        case 'h':
            fprintf(stderr,
                    "%s\n"
                    "\t-h: print this message\n"
                    "\t-i: initial conditions (%s)\n"
                    "\t-o: output file name (%s)\n"
                    "\t-n: number of cells per side (%d)\n"
                    "\t-w: domain width in cells (%g)\n"
                    "\t-f: time between frames (%g)\n"
                    "\t-F: number of frames (%d)\n",
                    argv[0], ic.c_str(), fname.c_str(), 
                    nx, width, ftime, frames);
            return -1;
        case 'i':  ic     = optarg;          break;
        case 'o':  fname  = optarg;          break;
        case 'n':  nx     = atoi(optarg);    break;
        case 'w':  width  = atof(optarg);    break;
        case 'f':  ftime  = atof(optarg);    break;
        case 'F':  frames = atoi(optarg);    break;
        default:
            fprintf(stderr, "Unknown option (-%c)\n", c);
            return -1;
        }
    }

    //TODO: Add 3 initializations
    void (*icfun1)(Sim::real& h, double x, double y) = dam_break_h;
    void (*icfun2)(Sim::real& hu, double x, double y) = dam_break_hu;
    void (*icfun3)(Sim::real& hv, double x, double y) = dam_break_hv;
    if (ic == "dam_break") {
        icfun1 = dam_break_h;
        icfun2 = dam_break_hu;
        icfun3 = dam_break_hv;
    } else if (ic == "pond") {
        icfun1 = pond_h;
        icfun2 = pond_hu;
        icfun3 = pond_hv;
    } else if (ic == "river") {
        icfun1 = river_h;
        icfun2 = river_hu;
        icfun3 = river_hv;
    } else if (ic == "wave") {
        icfun1 = wave_h;
        icfun2 = wave_hu;
        icfun3 = wave_hv;
    } else {
        fprintf(stderr, "Unknown initial conditions\n");
    }
    
    Sim sim(width,width, nx,nx);
    SimViz<Sim> viz(fname.c_str(), sim);
    sim.init(icfun1, icfun2, icfun3);
    sim.solution_check();
    viz.write_frame();

    std::ofstream time_file;
    time_file.open("timings.csv");
    for (int i = 0; i < frames; ++i) {
#ifdef _OPENMP
        double t0 = omp_get_wtime();
        sim.run(ftime);
        double t1 = omp_get_wtime();
        printf("Time: %e\n", t1-t0);
        time_file << t1-t0 << std::endl;
#else
        sim.run(ftime);
#endif
        sim.solution_check();
        viz.write_frame();
    }
    time_file.close();
}
