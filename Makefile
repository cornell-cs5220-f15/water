CXX=icc
CXXFLAGS=-std=c++14
PYTHON=python

shallow: driver.cc central2d.h shallow2d.h minmod.h meshio.h
	$(CXX) $(CXXFLAGS) -o $@ $<

mac:
	make CXX=g++ CXXFLAGS=-std=c++14

run: dam_break.gif

dam_break.gif: dam_break.out
	$(PYTHON) visualizer.py dam_break.out dam_break.gif dam_break.png

wave.gif: wave.out
	$(PYTHON) visualizer.py wave.out wave.gif wave.png

dam_break.mp4: dam_break.out
	$(PYTHON) visualizer.py dam_break.out dam_break.mp4 dam_break.png

wave.mp4: wave.out
	$(PYTHON) visualizer.py wave.out wave.mp4 wave.png

dam_break.out: shallow
	./shallow -i dam_break -o dam_break.out

wave.out: shallow
	./shallow -i wave -o wave.out -F 100

shallow.pdf: intro.md shallow.md
	pandoc --toc $^ -o $@

shallow.md: shallow2d.h minmod.h central2d.h meshio.h driver.cc
	ldoc $^ -o $@

clean:
	rm -f shallow
	rm -f dam_break.* wave.*
	rm -f shallow.md shallow.pdf

