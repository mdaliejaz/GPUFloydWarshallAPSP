#include<stdio.h>
#include<math.h>

int main(int argc, char** argv) 
{

	float *d_a, *d_b,*d_c;

	size_t pitch;
	int row=0;
	int i = 4;
	while (1)
	{
		row = pow(2,i);
		cudaMallocPitch(&d_a, &pitch, row * sizeof(float), row);
		
		if(!d_a)
		{
			printf("memory failed for 2^%d\n",i);
			return 1;
		}  
  		
		cudaMallocPitch(&d_b, &pitch, row * sizeof(float), row);  
		
		if(!d_b)
                { 
                        printf("memory failed for 2^%d\n",i);
                        	cudaFree(d_a);
			return 1;
               	}
 		
		cudaMallocPitch(&d_c, &pitch, row * sizeof(float), row);  

		 if(!d_c)
                { 
                        printf("memory failed for 2^%d\n",i);
                        cudaFree(d_a); cudaFree(d_b);
			return 1;
               	}

		printf("memory alloted for 2^%d x 2^%d\n",i,i);
		++i;
	}
}
