#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "size.h"

void main(){
	double * A = (double *) malloc(iters * sizeof(double));
	for(int i = 0; i < iters; i ++){
		A[i] = (double)i;	
	} 
	double sum = 0;
    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
	double tstart = omp_get_wtime();
	for(int i = 0; i < iters; i+=4){
		sum1 += sin(A[i  ]);
		sum2 -= sin(A[i+1]);
		sum3 += cos(A[i+2]);
		sum4 -= cos(A[i+3]);
	}
    sum = sum1 + sum2 + sum3 + sum4;
	tstart = omp_get_wtime() - tstart;
	printf("Serial\n%f, %f\n", sum, tstart);
}
