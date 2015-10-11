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

PLATFORM=mac
include Makefile.in.$(PLATFORM)

# ===
# Main driver and sample run

lshallow: ldriver.cc central2d.h shallow2d.h minmod.h meshio.h
	$(CXX) $(CXXFLAGS) $(LUA_CFLAGS) -o $@ $< $(LUA_LIBS)

.PHONY: run big
run: dam_break.gif

big: lshallow
	# ./shallow -i wave -o wave.out -n 1000 -F 100


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
	./lshallow tests.lua dam

wave.out: lshallow
	./lshallow tests.lua wave

# ===
# Generate documentation

shallow.pdf: intro.md shallow.md
	pandoc --toc $^ -o $@

shallow.md: shallow2d.h minmod.h central2d.h meshio.h ldriver.cc
	ldoc $^ -o $@

# ===
# Clean up

.PHONY: clean
clean:
	rm -f lshallow
	rm -f dam_break.* wave.*
	rm -f shallow.md shallow.pdf
	rm -rf *.dSYM

