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
