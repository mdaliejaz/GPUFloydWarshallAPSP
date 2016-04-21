#include<stdio.h>
#include<math.h>
#include <sys/time.h>

int rowSize;

__global__ void printGpu(float *d_a, int size)
{
	int i,j;
        for(i=0;i<size;i++)
        {
                for(j=0;j<size;j++)
                        printf("%0.1f\t", d_a[i*size+j]);
               	printf("\n");
        }
}

__global__ void Bloop_FW(float *d_a, int colSize)
{

	__shared__ int intermed;
	int col =  blockIdx.x * blockDim.x + threadIdx.x;
	if(col >= colSize)
		return;
	
	for(int k=0;k<colSize;k++)
        {
           for(int i = 0; i < colSize;i++)
           {

		if (threadIdx.x == 0) {
			intermed = d_a[i*colSize+k];
		}
		__syncthreads();

		d_a[i*colSize +	col ]  = fmin(intermed + d_a[k*colSize + col], d_a[i*colSize+col ]);
	}
	}	
}

void print_matrix(float *d,int size)
{
	int i,j;
	for(i=0;i<size;i++)	
	{
		for(j=0;j<size;j++)
			printf("%0.1f\t", d[i*size+j]);
		puts("");
	}
}

int main(int argc, char** argv) 
{
	float *d_a;
	float *a;	
	size_t pitch;
	rowSize = atoi(argv[1]);
	int colSize = rowSize;
	int i,j;
	cudaError_t err = cudaSuccess;  
	size_t totalSize = rowSize*colSize*sizeof(float);	

	a = (float *) malloc(totalSize);
	if (!a)
	{
		printf("Unable to allocate memory for host array\n");
		return 1;
	}	
	
	err = cudaMallocPitch(&d_a, &pitch, rowSize * sizeof(float), colSize);
	if(!d_a)
	{
		printf("memory failed for cudaMalloc");
		return 1;
	}  
  	
	if(err !=0){
        	printf("%s-%d",cudaGetErrorString(err),3);
   		return 1;  
    	}   

	for(i = 0; i < rowSize;i++)
		for (j=0;j<colSize;j++)
		{
			if (i == j){
				a[i*rowSize+j] = 0;
			}
			else {

				a[i*rowSize+j] = (i+j)%5? (i+j) : (i+j)%7;
			}
		}
 
	//puts("input matrix :");
	//print_matrix(a,rowSize);
	
	err = cudaMemcpy(d_a, a, totalSize, cudaMemcpyHostToDevice);
	if(err !=0){
        	printf("after h2d %s-%d",cudaGetErrorString(err),3);
        	return 1;
    	}   
	
	int threadsPerBlock;
        if (rowSize < 1024) {
                threadsPerBlock = rowSize;
        } else {
                threadsPerBlock = 1024;
        }
        int blocksPerGrid = rowSize/threadsPerBlock;

        struct timeval  tv1, tv2;
        gettimeofday(&tv1, NULL);

	/*for(k=0;k<rowSize;k++)
	{
	   for(i = 0; i < rowSize;i++)
	   {
               	Bloop_FW<<<blocksPerGrid, threadsPerBlock>>>(d_a,i,k,rowSize);
		cudaThreadSynchronize();	
	   }
	}*/
	Bloop_FW<<<blocksPerGrid, threadsPerBlock>>>(d_a,rowSize);
                cudaThreadSynchronize();	


	gettimeofday(&tv2, NULL);
        printf ("Total Execution time = %f seconds\n", (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec));

	printf("error = %s\n", cudaGetErrorString(cudaGetLastError()));
       	err = cudaMemcpy(a, d_a, totalSize, cudaMemcpyDeviceToHost);
	if(err !=0){
        	printf("final %s-%d",cudaGetErrorString(err),3);
        	return 1; 
    	}   

	puts("output matrix :");	
	print_matrix(a,rowSize);

	free(a);
	cudaFree(d_a);
	return 0;
}
