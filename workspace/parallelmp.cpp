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
	double sum = 0;
	double totalsum = 0;
	double tstart = omp_get_wtime();
	#pragma omp parallel private(sum) shared(totalsum)
	{
		#pragma omp sections
		{
			#pragma omp section
			{
				#pragma omp parallel
				{
					double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
					#pragma omp for
					for(int i = 0; i < iters; i +=4){
						#pragma vector aligned
						#pragma ivdep
						sum1 += A[i];
						sum2 -= A[i+1];
						sum3 += A[i+2];
						sum4 -= A[i+3];
					}
					sum = sum1 + sum2 + sum3 + sum4;
				}
			}

			#pragma omp section
			{
				#pragma omp atomic
				totalsum += sum;
			}
			
		}
	}
	tstart = omp_get_wtime() - tstart;

	printf("\n%f, %f\n", totalsum, tstart);
}
