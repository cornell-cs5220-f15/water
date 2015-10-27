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

    void (*icfun)(Sim::vec& u, double x, double y) = dam_break;
    if (ic == "dam_break") {
        icfun = dam_break;
    } else if (ic == "pond") {
        icfun = pond;
    } else if (ic == "river") {
        icfun = river;
    } else if (ic == "wave") {
        icfun = wave;
    } else {
        fprintf(stderr, "Unknown initial conditions\n");
    }
    
    Sim sim(width,width, nx,nx);
    SimViz<Sim> viz(fname.c_str(), sim);
    sim.init(icfun);
    sim.solution_check();
    viz.write_frame();

    int nproc=4;
    int nprocx=2;
    int nprocy=2;
    double cfl=0.45;
    double dx=width/nx;
    double t0 , t1;
    double cx, cy;
    t0 = omp_get_wtime();
#pragma omp parallel num_threads(nproc)
  {

    Sim sim_(width/nprocx,width/nprocy, nx/nprocx,nx/nprocy);

    int rankx,ranky;
    int rank = omp_get_thread_num();
    ranky = rank/nprocx;
    rankx = rank%nprocx;
 
    sim_.init2(sim,rankx,ranky,nprocx,nprocy);
    for (int i = 0; i < frames; ++i) {
    
      bool done = false;
      double t = 0;
      while (!done) {
          double dt;
          for (int io = 0; io < 2; ++io) {
#pragma omp master
              {
                sim.apply_periodic();
              }

#pragma omp barrier
           
              sim_.copyfrom(sim,rankx,ranky,nprocx,nprocy);

#pragma omp parallel reduction(max : cx,cy)
              {
                sim_.compute_fg_speeds(cx, cy);
              }
              sim_.limited_derivs();
              if (io == 0) {
                  dt = cfl / std::max(cx/dx, cy/dx);
                  if (t + 2*dt >= ftime) {
                      dt = (ftime-t)/2;
                      done = true;
                  }
              }
              sim_.compute_step(io, dt);
              t += dt;
              sim_.copyto(sim,rankx,ranky);
          }

      }
    

#pragma omp master
      {
        sim.solution_check();
        viz.write_frame();
      }

    }

  }    

  t1 = omp_get_wtime();
  printf("Time: %e\n", t1-t0);
}
