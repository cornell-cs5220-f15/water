CXX=icc
CXXFLAGS=-std=c++14

shallow: driver.cc central2d.h shallow2d.h minmod.h
	$(CXX) $(CXXFLAGS) -o $@ $<

shallow0: shallow.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

mac:
	make CXX=g++ CXXFLAGS=-std=c++14

run: shallow
	./shallow
	convert test.pgm test.png
	convert test2.pgm test2.png

shallow.pdf: intro.md shallow.md
	pandoc --toc $^ -o $@

shallow.md: shallow2d.h minmod.h central2d.h driver.cc
	ldoc $^ -o $@

clean:
	rm -f shallow test.pgm test2.pgm test.png test2.png
