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

PLATFORM=icc
include Makefile.in.$(PLATFORM)

SIMULATORS = \
	shallow \
	shallow_block \
	shallow_block_par\
	shallow_buggy \
	shallow_copy \
	shallow_par \
	shallow_vec \

HEADERS = \
	central2d.h \
	central2d_block.h \
	central2d_block_par.h\
	central2d_buggy.h \
	central2d_copy.h \
	central2d_par.h \
	central2d_vec.h \
	meshio.h \
	minmod.h \
	shallow2d.h \
	shallow2d_block.h \
	shallow2d_vec.h \

# ===
# Main driver and sample run
.PHONY: all run run_% big

all: $(SIMULATORS)

shallow: driver.cc central2d.h shallow2d.h minmod.h meshio.h
	$(CXX) $(CXXFLAGS) -o $@ $< -DVERSION_ref

shallow-timing: driver.cc central2d.h shallow2d.h minmod.h meshio.h
	$(CXX) $(CXXFLAGS) -DTIMING_ENABLED -o shallow $< -DVERSION_ref

shallow_%: driver.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< -DVERSION_$*

shallow-timing_%: driver.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -DTIMING_ENABLED -o shallow_$* $< -DVERSION_$*

run: shallow
	qsub run.pbs -N shallow -vARG1=shallow

run-ampl: shallow
	qsub run-ampl.pbs -N shallow-ampl -vARG1=shallow

run_%: shallow_%
	qsub run.pbs -N $* -vARG1=$<

run-ampl_%: shallow_%
	qsub run-ampl.pbs -N $*-ampl -vARG1=$<

time: clean $(shell echo shallow-timing{,_vec}) $(shell echo run{,_vec})

big: shallow
	./shallow -i wave -o wave.out -n 1000 -F 100

# ===
# Example analyses

.PHONY: maqao scan-build

maqao: shallow
	( module load maqao ; \
	  maqao cqa ./shallow fct=compute_step uarch=HASWELL )

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

dam_break.out: shallow
	./shallow -i dam_break -o dam_break.out

wave.out: shallow
	./shallow -i wave -o wave.out -F 100

# ===
# Generate documentation

shallow.pdf: intro.md shallow.md
	pandoc --toc $^ -o $@

shallow.md: shallow2d.h minmod.h central2d.h meshio.h driver.cc
	ldoc $^ -o $@

# ===
# Rules for printing

print-%: ; @echo $*=$($*)

# ===
# Clean up

.PHONY: clean
clean:
	rm -f $(SIMULATORS)
	rm -f dam_break.* wave.*
	rm -f shallow.md shallow.pdf
