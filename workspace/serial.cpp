#include <stdio.h>
#include <omp.h>
#include "size.h"


void main(){
	double * A = (double *) malloc(iters * sizeof(double));
	for(int i = 0; i < iters; i ++){
		A[i] = (double)i;	
	} 
	double sum = 0;
	double tstart = omp_get_wtime();
	for(int i = 0; i < iters; i ++){
		sum += A[i];
		sum -= A[i];
		sum += A[i];
		sum -= A[i];
	}
	tstart = omp_get_wtime() - tstart;
	printf("\n%f, %f\n", sum, tstart);

}
