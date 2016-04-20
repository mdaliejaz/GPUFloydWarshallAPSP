#include<stdio.h>
#include<math.h>


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

__global__ void Bloop_FW(float *d_a,int i,int k, int colSize)
{

	__shared__ int intermed;
	int col =  blockIdx.x * blockDim.x + threadIdx.x;
	if(col >= colSize)
		return;

	if (threadIdx.x == 0) {
		intermed = d_a[i*colSize+k];
		//d_a[i*colSize + threadIdx.x ]  = ((intermed + d_a[k*colSize + threadIdx.x]) < d_a[i*colSize + threadIdx.x ]) ? (intermed + d_a[k*colSize + threadIdx.x]) : (d_a[i*colSize + threadIdx.x ]);
		//d_a[i*colSize + col ]  = fmin(intermed + d_a[k*colSize + col] ,  d_a[i*colSize+col ]); 	
	}
	__syncthreads();

	d_a[i*colSize +	col ]  = fmin(intermed + d_a[k*colSize + col], d_a[i*colSize+col ]);
	//d_a[i*colSize + threadIdx.x ]  = ((intermed + d_a[k*colSize + threadIdx.x]) < d_a[i*colSize + threadIdx.x ]) ? (intermed + d_a[k*colSize + threadIdx.x]) : (d_a[i*colSize + threadIdx.x ]);


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
	rowSize = 8192;
	int colSize = rowSize;
	int i,j,k;
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
        getchar();  
    	}   
	

	int threadsPerBlock;
	int noOfBlocks;

	if (rowSize < 1024)
        { 
                threadsPerBlock	= rowSize;
       	}
	else
	{
		threadsPerBlock = 1024;
	}

        noOfBlocks = rowSize / threadsPerBlock;


	for(k=0;k<rowSize;k++)
	{
	   for(i = 0; i < rowSize;i++)
	   {
                	Bloop_FW<<<noOfBlocks,threadsPerBlock>>>(d_a,i,k,rowSize);
			cudaThreadSynchronize();
	   }
	}

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
