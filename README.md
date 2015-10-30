# Water #
This directory contains multiple optimized shallow water simulators.

## Building ##
This repository contains many versions of a shallow water simulator. For
example, [`central2d.h`](central2d.h) is an unoptimized simulator,
[`central2d_vec.h`](central2d_vec.h) is a vectorized simulator, and
[`central2d_par.h`](central2d_par.h) is a parallelized simulator. All the
simulators share the same driver, [`driver.cc`](driver.cc), and the driver
decides which simulator to instantiate and use via a set of command line flags.
The details of this mechanism are abstracted away by the Makefile; all you have
to know is that if you want to run version `v` of the simulator, run `make
shallow_v`. For example, to build the vectorized simulator, run

```bash
make shallow_vec
```

By default `driver.cc` will run a simulator and check its correctness against a
reference implementation. If you want to bypass the correctness checks and time
the simulator, you should run `make shallow-timing_v`. For example, to build a
timed version of the vectorized simulator, run

```bash
make shallow-timing_vec
```

## Running ##
To run simulator version `v` on the cluster, run

```
make run_v
```

To run simulator version `v` with VTune Amplifier, run

```
make run-ampl_v
```

## Plotting ##
We plot timing with our own python script [`plotter.py`](plotter.py). This
generates cells/seconds versus nx (number of cells per side).

After you get `.o` files from qsub, you can run this script like below. This
will include every files with `.o`.

```
./plotter.py -t 1 *.o[1-9]*
```

This will automatically find timing in `.o` files and generate `.cvs` files,
and sort by nx.

You can also generate for two different version in a single plot. For example,
if you have `vec.o12345` and `par.o12346`, run like below:

```
./plotter.py -t 1 vec.o12345 par.o12345
```

If you want to generate different types of graphs.
```
./plotter.py -t 2 vec.o12345 par.o12345
```

Currently, type 1 draws "cells/seconds vs nx", and type 2 draws "seconds vs
nx". To see more infomation type:

```
./plotter.py -h
```

