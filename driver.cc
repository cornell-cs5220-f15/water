#include "central2d.h"
#include "central2d_buggy.h"
#include "central2d_copy.h"
#include "central2d_par.h"
#include "central2d_vec.h"
#include "meshio.h"
#include "minmod.h"
#include "shallow2d.h"
#include "shallow2d_vec.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
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

// As we optimize the wave equation simulator, we develop many different
// versions of the code. For example, me may modify the limiter to perform the
// minmod operation quicker, or we may modify the central2d solver to run in
// parallel. Despite having many different versions of the code, we need a way
// to run any one version easily. This macro magic is a way to do that.
//
// During compile time, we pass in a macro of the form `VERSION_<version>`
// where `<version>` determines the version of the simulator to use. For
// example, if we compile driver.cc like this
//
//     icc -o shallow_foo driver.cc -DVERSION_foo
//
// then the foo version of the simulator will be used. If no version is
// specified, the code does not compile.
typedef Central2D<Shallow2D, MinMod<Shallow2D::real>> ReferenceSim;

#if defined(VERSION_ref)
    typedef ReferenceSim Sim;
#elif defined(VERSION_vec)
    typedef Central2DVec<Shallow2DVec, MinMod<Shallow2DVec::real>> Sim;
#elif defined(VERSION_buggy)
    typedef Central2DBuggy<Shallow2D, MinMod<Shallow2D::real>> Sim;
#elif defined(VERSION_copy)
    typedef Central2DCopy<Shallow2D, MinMod<Shallow2D::real>> Sim;
#elif defined(VERSION_par)
    typedef Central2DPar<Shallow2D, MinMod<Shallow2D::real>> Sim;
#else
    static_assert(false, "Please define a valid VERSION_* macro.");
#endif

// When TIMING_ENABLED is defined, simulators are timed, rather than checked
// for correctness.
#define TIMING_ENABLED

/*
 * `validate(ref_sim, sim)` validates that `ref_sim` and `sim` are simulating
 * equal sized grids and that the two grids are equal. If the simulators have
 * unequal sized grids or have unequal grids, an error message is printed to
 * stderr and the program is exited with error.
 */
void validate(const ReferenceSim& ref_sim, const Sim& sim) {
    // Check that the two grids are of equal size.
    if (ref_sim.xsize() != sim.xsize() || ref_sim.ysize() != sim.ysize()) {
        fprintf(
            stderr,
            "reference simulator size (%d x %d) != simulator size (%d, %d).\n",
            ref_sim.xsize(), ref_sim.ysize(), sim.xsize(), sim.ysize()
        );
        exit(-1);
    }

    // Check that the two grids are component-wise equal.
    bool different = false;
    for (int x = 0; x < ref_sim.xsize(); ++x) {
        for (int y = 0; y < ref_sim.ysize(); ++y) {
            auto& ref_vec = ref_sim(x, y);
            auto& vec = sim(x, y);
            for (int m = 0; m < ref_vec.size(); ++m) {
                // The check that `ref_vec[m] == vec[m]` will often fail even
                // when the two are in fact equal! We allow them to differ by
                // some small amount. There is nothing special about `10 *
                // FLT_EPSILON`; it is an arbitrary tolerance.
                if (abs(ref_vec[m] - vec[m]) > 10 * FLT_EPSILON) {
                    fprintf(
                        stderr,
                        "reference simulator(%d, %d)[%d] = %f != "
                        "simulator(%d, %d)[%d] = %f.\n",
                        x, y, m, ref_vec[m], x, y, m, vec[m]
                    );
                    different = true;
                }
            }
        }
    }
    if (different) {
        exit(-1);
    }
}


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
template <class Sim>
void dam_break(typename Sim::vec& u, double x, double y)
{
    x -= 1;
    y -= 1;
    u[0] = 1.0 + 0.5*(x*x + y*y < 0.25+1e-5);
    u[1] = 0;
    u[2] = 0;
}

// Still pond (ideally, nothing should move here!)
template <class Sim>
void pond(typename Sim::vec& u, double x, double y)
{
    u[0] = 1.0;
    u[1] = 0;
    u[2] = 0;
}

// River (ideally, the solver shouldn't do much with this, either)
template <class Sim>
void river(typename Sim::vec& u, double x, double y)
{
    u[0] = 1.0;
    u[1] = 1.0;
    u[2] = 0;
}


// Wave on a river -- develops a shock in finite time!
template <class Sim>
void wave(typename Sim::vec& u, double x, double y)
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
    std::string fname = "waves.out";
    std::string ic = "dam_break";
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

    auto icfun     = dam_break<Sim>;
    auto ref_icfun = dam_break<ReferenceSim>;
    if (ic == "dam_break") {
        icfun = dam_break<Sim>;
        ref_icfun = dam_break<ReferenceSim>;
    } else if (ic == "pond") {
        icfun = pond<Sim>;
        ref_icfun = pond<ReferenceSim>;
    } else if (ic == "river") {
        icfun = river<Sim>;
        ref_icfun = river<ReferenceSim>;
    } else if (ic == "wave") {
        icfun = wave<Sim>;
        ref_icfun = wave<ReferenceSim>;
    } else {
        fprintf(stderr, "Unknown initial conditions\n");
    }

    // Print parameters for plotting
    printf("\nparse_line\n");
    printf("ic: %s\n", ic.c_str());
    printf("fname: %s\n", fname.c_str());
    printf("nx: %d\n", nx);
    printf("width: %f\n", width);
    printf("ftime: %f\n", ftime);
    printf("frames: %d\n\n", frames);

    // Initialize simulator
    Sim sim(width, width, nx, nx);
    SimViz<Sim> viz(fname.c_str(), sim);
    sim.init(icfun);
    sim.solution_check();
    viz.write_frame();

    #ifdef TIMING_ENABLED
        printf("timing!\n");
        #ifndef _OPENMP
            // timing requires OpenMP
            assert(false);
        #endif
        double t0 = omp_get_wtime();
        for (int i = 0; i < frames; ++i) {
            sim.run(ftime);
        }
        double t1 = omp_get_wtime();
        printf("Time: %e\n", (t1 - t0) / frames);
    #else
        printf("not timing!\n");
        // Initialize reference simulator
        ReferenceSim ref_sim(width, width, nx, nx);
        ref_sim.init(ref_icfun);
        ref_sim.solution_check();

        // Reference check for only one time, so no timing it
        for (int i = 0; i < frames; ++i) {
            #ifdef _OPENMP
                double t0 = omp_get_wtime();
                sim.run(ftime);
                double t1 = omp_get_wtime();
                printf("Time: %e\n", t1-t0);
            #else
                sim.run(ftime);
            #endif
            ref_sim.run(ftime);
            sim.solution_check();
            ref_sim.solution_check();
            viz.write_frame();
            validate(ref_sim, sim);
        }
    #endif
}
