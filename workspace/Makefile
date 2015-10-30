CC = icc
CFLAGS = -g -O3 -openmp
PCFLAGS = -axCORE-AVX512,CORE-AVX2 -offload-attribute-target=mic 

TARGETS = serial parallel parallelmp

all: $(TARGETS)

serial: size.h serial.cpp
	$(CC) $(CFLAGS) -o serial serial.cpp 

parallel: size.h parallel.cpp 
	$(CC) $(PCFLAGS) $(CFLAGS) -o parallel parallel.cpp 

parallelmp: size.h parallelmp.cpp
	$(CC) $(PCFLAGS) $(CFLAGS) -o parallelmp parallelmp.cpp

clean: 
	rm -rf $(TARGETS) *.o*

run:
	qsub job-parallel
	qsub job-parallelmp
	qsub job-serial

