#include <stdio.h>
#include <math.h>
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
            double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
            #pragma omp for
            for (int i = 0; i < iters; i += 4) {
                sum1 += sin(A[i  ]);
                sum2 -= sin(A[i+1]);
                sum3 += cos(A[i+2]);
                sum4 -= cos(A[i+3]);
            }
            sum = sum1 + sum2 + sum3 + sum4;

            #pragma omp atomic
            totalsum += sum;
        }
    }
	tstart = omp_get_wtime() - tstart;

	printf("Parallel Offload\n%f, %f\n", totalsum, tstart);
}
