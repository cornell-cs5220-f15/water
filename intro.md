% Shallow water simulation
% David Bindel
% 2015-09-30

# Introduction

## The Fifth Dwarf

In an 2006 report on ["The Landscape of Parallel Computing"][view],
a group of parallel computing researchers at Berkeley suggested
that high-performance computing platforms be evaluated with respect
to "13 dwarfs" -- frequently recurring computational patterns in
high-performance scientific code.  This assignment represents the
fifth dwarf on the list: structured grid computations.  We have
already seen one example of structured grid computations in class with
the Game of Life, but this pattern is common in many areas of physical
simulation.  It features high spatial locality and allows regular
access patterns, and is in principal one of the easier types of
computations to parallelize.

Structured grid computations are particularly common in fluid dynamics
simulations, and the code that you will tune in this assignment is an
example of such a simulation.  You will be optimizing and
parallelizing a finite volume solver for the shallow water equations,
a two-dimensional PDE system that describes waves that are very long
compared to the water depth.  This is an important system of equations
that applies even in situations that you might not initially think of
as "shallow"; for example, tsunami waves are long enough that they can
be modeled using the shallow water equations even when traveling over
mile-deep parts of oceans.  There is also a very readable
[Wikipedia article][wiki] on the shallow water equations, complete
with a little animation similar to the one you will be producing.  I
was inspired to use this system for our assignment by reading the
chapter on [shallow water simulation in MATLAB][exm] from Cleve
Moler's books on "Experiments in MATLAB" and then getting annoyed that
he chose a method with a stability problem.

[view]: http://www.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf
[exm]: https://www.mathworks.com/moler/exm/chapters/water.pdf
[wiki]: https://en.wikipedia.org/wiki/Shallow_water_equations


## Your mission

You are provided with the following performance-critical C++ files:

- `shallow2d.h` -- an implementation of shallow water physics
- `minmod.h` -- a (possibly efficient)  MinMod limiter
- `central2d.h` -- a finite volume solver for 2D hyperbolic PDE

In addition, you are given the following codes for running the
simulation and getting pretty pictures out:

- `meshio.h` -- I/O routines
- `driver.cc` -- a driver file that runs the simuation
- `visualizer.py` -- Python visualization script

For this assignment, you should attempt three tasks:

1.  *Profiling*:  The current code is not particularly tuned, and there
    are surely some bottlenecks.  Profile the computation and
    determine what parts of the code are slowest.  I encourage you to
    use profiling tools (e.g. VTune Amplifier), but you may also
    manually instrument the code with timers.

2.  *Parallelization*: You should parallelize your code using OpenMP,
    and study the speedup versus number of processors on both the main
    cores on the nodes and on the Xeon Phi boards.  Set up both strong
    and weak scaling studies, varying the number of threads you
    employ.  You may start with a naive parallelization
    (e.g. parallelizing the for loops in the various subroutines), but
    this is not likely to give you the best possible performance,
    particularly on the Phi.

3.  *Tuning*:  You should tune your code in order to get it to run as
    fast as possible.  This may involve a domain decomposition
    with per-processor ghost cells and batching of time steps between
    synchronization; it may involve vectorization of the computational
    kernels; or it may involve eliminating redundant computations in
    the current implementation.

The primary deliverable for your project is a report that describes
your performance experiments and attempts at tuning, along with what
you learned about things that did or did not work.  Good things for
the report include:

- Profiling results
- Speedup plots and scaled speedup plots
- Performance models that predict speedup

In addition, you should also provide the code, and ideally scripts
that make it simple to reproduce any performance experiments you've
run.

## Tuning readings

You are welcome to read about the shallow water equations and about
finite volume methods if you would like, and this may help you in
understanding what you're seeing.  But it is possible to tune your
codes without understanding all the physics behind what is going on!
I recommend two papers in particular that talk about tuning of finite
difference and finite volume codes on accelerators: one on
[optimizing a 3D finite difference code][3dfd] on the Intel cores and
the Xeon Phi, and one on
[optimizing a shallow water simulator][brodtkorb]
on NVidia GPUs.  The GPU architecture is different, but some of the
concepts should transfer to thinking about improving performance on
the Phis.

[3dfd]: https://software.intel.com/en-us/articles/eight-optimizations-for-3-dimensional-finite-difference-3dfd-code-with-an-isotropic-iso
[brodtkorb]: http://cmwr2012.cee.illinois.edu/Papers/Special%20Sessions/Advances%20in%20Heterogeneous%20Computing%20for%20Water%20Resources/Brodtkorb.Andre_R.pdf

## Logistical notes

### Timeline

As with the previous assignment, this assignment involves two stages.
You have just over three weeks, and should work in teams of three.
After two weeks (Oct 19), you should submit your initial report (and
code) for peer review; reviews are due by Oct 22.  Final reports are
due one week later (Oct 27).  I hope many of you will wrap up before
that; the third project should be out by Oct 22.

### Peer review logistics

Since the first assignment, GitHub has added a feature to
[attach PDF files to issues and pull request comments][pdf].  You
should take advantage of this feature to submit your review as a
comment on the pull request for the group you are reviewing.
You should still look at the codes from the other groups, though!

[pdf]: https://github.com/blog/2061-attach-files-to-comments

### Notes on the documentation

The documentation for this project is generated automatically from
structured comments in the source files using a simple tool called
[ldoc][ldoc] that I wrote some years ago.  You may or may not choose
to use [ldoc][ldoc] for your version.

[ldoc]: https://github.com/dbindel/ldoc

### Notes on C++ usage

The reference code I've given you is in C++.  I wanted to use C, but
was persuaded that I could write a clearer, cleaner implementation in
C++ -- and an implementation that will ultimately be easier for you to
tune.

While I have tried not to do anything too obscure, this code does use
some C++ 11 features (e.g. the `constexpr` notation used to tell the
compiler something is a compile-time constant).  If you want to build
on your own machine, you may need to figure out the flag needed to
tell your compiler that you are using this C++ dialect.

