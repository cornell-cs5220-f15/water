#include "central2d.h"
#include "shallow2d.h"
#include "minmod.h"


//ldoc on
/**
 * # Driver routines
 * 
 * For the driver, we need to put everything together: we're running
 * a `Central2D` solver for the `Shallow2D` physics with a `MinMod`
 * limiter.
 */

typedef Central2D< Shallow2D, MinMod<Shallow2D::real> > Sim;

/**
 * ## Initial states and graphics
 * 
 * The following functions define some interesting initial conditions.
 * Ideally, I would be doing this via a Python interface.  But I
 * couldn't be bothered to deal with the linker.
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


/**
 * ## Summary plots
 * 
 * We can plot either height or (total) momentum as interesting
 * scalar quantities.  The ranges (0 to 3.0 and 0 to 2.5) are
 * completely made up -- it would probably be smarter to change
 * those!
 */

int show_height(const Sim::vec& u)
{
    return 255 * (u[0] / 3.0);
}

int show_momentum(const Sim::vec& u)
{
    return 255 * sqrt(u[1]*u[1] + u[2]*u[2]) / 2.5;
}


/**
 * # Main driver
 * 
 * Again, this should really invoke an option parser, or be glued
 * to an interface in some appropriate scripting language (Python,
 * or perhaps Lua).
 */

int main()
{
    Sim sim(2,2, 200,200, 0.2);
    sim.init(dam_break);
    sim.solution_check();
    sim.write_pgm("test.pgm", show_height);
    sim.run(0.5);
    sim.solution_check();
    sim.write_pgm("test2.pgm", show_height);
}
