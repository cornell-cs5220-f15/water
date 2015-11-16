# Jiang-Tadmor central difference scheme

[Jiang and Tadmor][jt] proposed a high-resolution finite difference
scheme for solving hyperbolic PDE systems in two space dimensions.
The method is particularly attractive because, unlike many other
methods in this space, it does not require that we write any
solvers for problems with special initial data (so-called Riemann
problems), nor even that we compute Jacobians of the flux
functions.

While this code is based loosely on the Fortran code at the end of
Jiang and Tadmor's paper, we've written the current code to be
physics-agnostic (rather than hardwiring it to the shallow water
equations -- or the Euler equations in the Jiang-Tadmor paper).
If you're interested in the Euler equations, feel free to add your
own physics class to support them!

[jt]: http://www.cscamm.umd.edu/tadmor/pub/central-schemes/Jiang-Tadmor.SISSC-98.pdf

## Staggered grids

The Jiang-Tadmor scheme works by alternating between a main grid
and a staggered grid offset by half a step in each direction.
Understanding this is important, particularly if you want to apply
a domain decomposition method and batch time steps between
synchronization barriers in your parallel code!

In even-numbered steps, the entry `u(i,j)` in the array of solution
values represents the average value of a cell centered at a point
$(x_i,y_j)$.  At the following odd-numbered step, the same entry
represents values for a cell centered at $(x_i + \Delta x/2, y_j +
\Delta y/2)$.  However, whenever we run a simulation, we always take
an even number of steps, so that outside the solver we can just think
about values on the main grid.  If `uold` and `unew` represent the
information at two successive *even* time steps (i.e. they represent
data on the same grid), then `unew(i,j)` depends indirectly on
`u(p,q)` for $i-3 \leq p \leq i+3$ and $j-3 \leq q \leq j+3$.

We currently manage this implicitly: the arrays at even time steps
represent cell values on the main grid, and arrays at odd steps
represent cell values on the staggered grid.  Our main `run`
function always takes an even number of time steps to ensure we end
up on the primary grid.

## MinMod limiter

Numerical methods for solving nonlinear wave equations are
complicated by the fact that even with smooth initial data, a
nonlinear wave can develop discontinuities (shocks) in finite time.

This makes for interesting analysis, since a "strong" solution
that satisfies the differential equation no longer makes sense at
a shock -- instead, we have to come up with some mathematically
and physically reasonable definition of a "weak" solution that
satisfies the PDE away from the shock and satisfies some other
condition (an entropy condition) at the shock.

The presence of shocks also makes for interesting *numerical*
analysis, because we need to be careful about employing numerical
differentiation formulas that sample a discontinuous function at
points on different sides of a shock.  Using such formulas naively
usually causes the numerical method to become unstable.  A better
method -- better even in the absence of shocks! -- is to consider
multiple numerical differentiation formulas and use the highest
order one that "looks reasonable" in the sense that it doesn't
predict wildly larger slopes than the others.  Because these
combined formulas *limit* the wild behavior of derivative estimates
across a shock, we call them *limiters*.  With an appropriate limiter,
we can construct methods that have high-order accuracy away from shocks
and are at least first-order accurate close to a shock.  These are
sometimes called *high-resolution* methods.

The MinMod (minimum modulus) limiter is one example of a limiter.
The MinMod limiter estimates the slope through points $f_-, f_0, f_+$
(with the step $h$ scaled to 1) by
$$
  f' = \operatorname{minmod}((f_+-f_-)/2, \theta(f_+-f_0), \theta(f_0-f_-))
$$
where the minmod function returns the argument with smallest absolute
value if all arguments have the same sign, and zero otherwise.
Common choices of $\theta$ are $\theta = 1.0$ and $\theta = 2.0$.

There are many other potential choices of limiters as well.  We'll
stick with this one for the code, but you should feel free to
experiment with others if you know what you're doing and think it
will improve performance or accuracy.


