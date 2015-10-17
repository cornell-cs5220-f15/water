#include <stdio.h>
#include <omp.h>
#include "size.h"

void main(){
	double *A;
	int foo;
	foo = posix_memalign((void**)&A, 64,iters*sizeof(double));
	for(int i = 0; i < iters; i ++){
		A[i] = (double)i;	
	} 
	double totalsum = 0;
	double tstart = omp_get_wtime();

    #pragma offload target(mic) in(A : length(iters))
    {
        #pragma omp parallel
        {
            double sum;
            #pragma omp for
            for (int i = 0; i < iters; i++) {
                sum += fn(A[i]);
            }

            #pragma omp atomic
            totalsum += sum;
        }
    }
	tstart = omp_get_wtime() - tstart;

	printf("Parallel Offload\n%f, %f\n", totalsum, tstart);
}
