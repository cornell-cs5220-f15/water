#include "central2d.h"
#include "central2d_v2.h"
#include "shallow2d.h"
#include "minmod.h"
#include "meshio.h"

#include <cmath>
#include <cstring>
#include <unistd.h>


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

#ifdef VERSION_v2
    typedef Central2DV2< Shallow2D, MinMod<Shallow2D::real> > Sim;
#else
    typedef Central2D< Shallow2D, MinMod<Shallow2D::real> > Sim;
#endif

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

// Still pond (ideally, nothing should move here!)
void pond(Sim::vec& u, double x, double y)
{
    u[0] = 1.0;
    u[1] = 0;
    u[2] = 0;
}

// River (ideally, the solver shouldn't do much with this, either)
void river(Sim::vec& u, double x, double y)
{
    u[0] = 1.0;
    u[1] = 1.0;
    u[2] = 0;
}


// Wave on a river -- develops a shock in finite time!
void wave(Sim::vec& u, double x, double y)
{
    using namespace std;
    u[0] = 1.0 + 0.2 * sin(M_PI*x);
    u[1] = 1.0;
    u[2] = 0;
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
    const char* fname = "waves.out";
    const char* ic = "dam_break";
    int    nx = 200;
    double width = 2.0;
    double ftime = 0.01;
    int    frames = 50;

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
                    argv[0], ic, fname, nx, width, ftime, frames);
            return -1;
        case 'i':  ic     = strdup(optarg);  break;
        case 'o':  fname  = strdup(optarg);  break;
        case 'n':  nx     = atoi(optarg);    break;
        case 'w':  width  = atof(optarg);    break;
        case 'f':  ftime  = atof(optarg);    break;
        case 'F':  frames = atoi(optarg);    break;
        default:
            fprintf(stderr, "Unknown option (-%c)\n", c);
            return -1;
        }
    }

    void (*icfun)(Sim::vec& u, double x, double y);
    if (strcmp(ic, "dam_break") == 0) {
        icfun = dam_break;
    } else if (strcmp(ic, "pond") == 0) {
        icfun = pond;
    } else if (strcmp(ic, "river") == 0) {
        icfun = river;
    } else if (strcmp(ic, "wave") == 0) {
        icfun = wave;
    } else {
        fprintf(stderr, "Unknown initial conditions\n");
    }

    Sim sim(width,width, nx,nx);
    SimViz<Sim> viz(fname, sim);
    sim.init(icfun);
    sim.solution_check();
    viz.write_frame();
    for (int i = 0; i < frames; ++i) {
        sim.run(ftime);
        sim.solution_check();
        viz.write_frame();
    }
}
