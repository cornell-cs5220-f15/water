CXX=icc
CXXFLAGS=-std=c++14

shallow: shallow.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

mac:
	make CXX=g++ CXXFLAGS=-std=c++14

run: shallow
	./shallow
	convert test.pgm test.png
	convert test2.pgm test2.png

shallow.pdf: shallow.md
	pandoc $< -o $@

shallow.md: shallow.cc
	ldoc $< -o $@

clean:
	rm -f shallow test.pgm test2.pgm test.png test2.png
