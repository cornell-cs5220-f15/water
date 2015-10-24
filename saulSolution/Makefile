#
# To build with a different compiler / on a different platform, use
#     make PLATFORM=xxx
#
# where xxx is
#     icc = Intel compilers
#     gcc = GNU compilers
#     clang = Clang compiler (OS X default)
#
# Or create a Makefile.in.xxx of your own!
#

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
PLATFORM=icc-mac
else
PLATFORM=icc
endif

include Makefile.in.$(PLATFORM)

# ===
# Main driver and sample run

lshallow: ldriver.o shallow2d.o stepper.o
	$(CC) $(CFLAGS) $(LUA_CFLAGS) -o $@ $^ $(LUA_LIBS)

ldriver.o: ldriverparal.c shallow2d.h
	$(CC) $(CFLAGS) $(LUA_CFLAGS) -c $<

shallow2d.o: shallow2d.c
	$(CC) $(CFLAGS) -c $<

stepper.o: stepperpar.c
	$(CC) $(CFLAGS) -c $<

lshallow.dSYM: lshallow
	dsymutil lshallow -o lshallow.dSYM

.PHONY: run big iprofile
run: dam_break.gif

.PHONY: iprofile
iprofile: lshallow lshallow.dSYM
	iprofiler -timeprofiler -o lshallow_perf ./lshallow testspar.lua dam 400
	open lshallow_perf.dtps

big: lshallow
	./lshallow testspar.lua dam 400


# ===
# Example analyses

.PHONY: maqao scan-build

maqao: lshallow
	( module load maqao ; \
	  maqao cqa ./lshallow fct=compute_step uarch=HASWELL )

scan-build:
	( module load llvm-analyzer ; \
	  scan-build -v --use-analyzer=/share/apps/llvm-3.7.0/bin/clang make )

# ===
# Generate visualizations (animated GIF or MP4)

dam_break.gif: dam_break.out
	$(PYTHON) visualizer.py dam_break.out dam_break.gif dam_break.png

wave.gif: wave.out
	$(PYTHON) visualizer.py wave.out wave.gif wave.png

dam_break.mp4: dam_break.out
	$(PYTHON) visualizer.py dam_break.out dam_break.mp4 dam_break.png

wave.mp4: wave.out
	$(PYTHON) visualizer.py wave.out wave.mp4 wave.png

# ===
# Generate output files

dam_break.out: lshallow
	./lshallow testspar.lua dam

wave.out: lshallow
	./lshallow testspar.lua wave

# ===
# Generate documentation

shallow.pdf: intro.md jt-scheme.md shallow.md
	pandoc --toc $^ -o $@

shallow.md: stepperpar.h stepperpar.c shallow2d.h shallow2d.c ldriverparal.c
	ldoc $^ -o $@

# ===
# Clean up

.PHONY: clean
clean:
	rm -f lshallow *.o
	rm -f dam_break.* wave.*
	rm -f shallow.md shallow.pdf
	rm -f *.optrpt
	rm -rf *.dSYM
	rm -rf lshallow_perf.dtps
