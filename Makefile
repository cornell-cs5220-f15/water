shallow: shallow.cc
	g++ -std=c++14 -o $@ $<

run: shallow
	./shallow
	convert test.pgm test.png
	convert test2.pgm test2.png

clean:
	rm -f test.pgm test2.pgm test.png test2.png
