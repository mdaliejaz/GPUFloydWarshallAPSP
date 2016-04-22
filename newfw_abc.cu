#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<limits.h>
#include<algorithm>
#include<sys/time.h>

using namespace std;

#define INF           INT_MAX-1

int tilesize[2] = {2, INT_MAX};
int rowSize;

void print_matrix(float *d)
{
	int i,j;
	for(i=0;i<32;i++)	
	{
		for(j=0;j<32;j++)
			printf("%0.1f\t", d[i*rowSize+j]);
		puts("");
	}
}

__global__ void Dloop_FW(float *d_a,int k, int rowSize)
{

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= rowSize)
		return;

	__shared__ int intermed;
        if (threadIdx.x == 0) {
               	intermed = d_a[rowSize*blockIdx.y + k];
       	}

       __syncthreads();

        d_a[blockIdx.y*rowSize + col]  = fmin(d_a[blockIdx.y*rowSize + col], intermed + d_a[k*rowSize+col]);

}

void FW_D_loop(float* d_a,int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int size)
{        
	int threadsPerBlock;

	if (size <= 1024)
		threadsPerBlock = size;
	else
		threadsPerBlock = 1024;

	dim3 blocksPerGrid( (rowSize + threadsPerBlock -1)/threadsPerBlock ,rowSize);

	for(int k = vRowStart; k < (vRowStart + size); k++)
	{
		        Dloop_FW<<<blocksPerGrid,threadsPerBlock>>>(d_a,k,rowSize);
			cudaThreadSynchronize();

	}
}

void DFW(float *d_a,int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int d)
{
	int r = tilesize[d];
	if (r >= currSize)
		FW_D_loop(d_a, xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize);
	else
	{
		int newsize = currSize/r;
		for(int k=1; k<=r; k++) {	
			for(int i=1; i<=r; i++) {
				for(int j=1; j<=r; j++) {
					DFW(d_a,(i-1)*newsize, (j-1)*newsize, (i-1)*newsize, (k-1)*newsize, (k-1)*newsize, (j-1)*newsize, newsize, d+1);
				}
			}	
		}
	}
}


__global__ void Cloop_FW(float *d_a,int vRowStart,int xColStart,int size, int rowSize)
{

	__shared__ int intermed;
	int col =  blockIdx.x * blockDim.x + threadIdx.x;
	if(col >= rowSize)
		return;
	

 	for(int k = vRowStart; k < (vRowStart + size); k++)
        {

                for(int j = xColStart; j < (xColStart + size); j++)
              {  
                  
			if (threadIdx.x == 0) 
				intermed = d_a[k*rowSize+j];
		
		__syncthreads();

        	d_a[col*rowSize + j ]  = fmin( d_a[col*rowSize + j ], d_a[col*rowSize + k] + intermed);
	   }
	}

}

void FW_C_loop(float *d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int size)
{        

	int threadsPerBlock;

        if (size <= 1024)
                threadsPerBlock = size;
        else
                threadsPerBlock = 1024;

        dim3 blocksPerGrid( (rowSize + threadsPerBlock -1)/threadsPerBlock ,rowSize);
	
	Cloop_FW<<<blocksPerGrid, threadsPerBlock>>>(d_a,vRowStart,xColStart,size,rowSize);
}



void CFW(float *d_a,int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int d)
{
	int r = tilesize[d];
	if (r >= currSize)
		FW_C_loop(d_a,xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize);
	else
	{
		int newsize = currSize/r;
		for(int k=1; k<=r; k++) {	
			for(int i=1; i<=r; i++) {
				CFW(d_a,(i-1)*newsize, (k-1)*newsize, (i-1)*newsize, (k-1)*newsize, (k-1)*newsize, (k-1)*newsize, newsize, d+1);
			}

			for(int i=1; i<=r; i++) {
				for(int j=1; j<=r; j++) {
					if(j != k)
						DFW(d_a,(i-1)*newsize, (j-1)*newsize, (i-1)*newsize, (k-1)*newsize, (k-1)*newsize, (j-1)*newsize, newsize, d+1);
				}
			}	
		}
	}
}


__global__ void Bloop_FW(float *d_a,int i,int k, int colSize)
{


	__shared__ int intermed;
//	if (threadIdx.x == k) {
		intermed = d_a[i*colSize+k];
//		d_a[i*colSize + threadIdx.x ]  = fmin(intermed + d_a[k*colSize + threadIdx.x] ,  d_a[i*colSize+threadIdx.x ]); 	
//	}

//	 __syncthreads();

	d_a[i*colSize +	threadIdx.x ]  = fmin(intermed + d_a[k*colSize + threadIdx.x], d_a[i*colSize+threadIdx.x ]);


}

void FW_B_loop(float *d_a,int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int size)
{        

	int threadsPerBlock;
	if (size < 1024)
        { 
                threadsPerBlock	= size;
       	}
	else
	{
		threadsPerBlock = 1024;
	}

	int noOfBlocks = rowSize / threadsPerBlock;

	for(int k = vRowStart; k < (vRowStart + size); k++)
	{
		for(int i = xRowStart; i < (xRowStart + size); i++)
		{
				Bloop_FW<<<noOfBlocks,threadsPerBlock>>>(d_a,i,k,rowSize);
				cudaThreadSynchronize();

		}
	}
}

void BFW(float* d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int d)
{
	int r = tilesize[d];
	if (r >= currSize)
		FW_B_loop(d_a,xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize);
	else
	{
		int newsize = currSize/r;
		for(int k=1; k<=r; k++) {
			for(int j=1; j<=r; j++) {
				BFW(d_a,(k-1)*newsize, (j-1)*newsize, (k-1)*newsize, (k-1)*newsize, (k-1)*newsize, (j-1)*newsize, newsize, d+1);
			}

			for(int i=1; i<=r; i++) {
				for(int j=1; j<=r; j++) {
					if(i != k)
						DFW(d_a,(i-1)*newsize, (j-1)*newsize, (i-1)*newsize, (k-1)*newsize, (k-1)*newsize, (j-1)*newsize, newsize, d+1);
				}
			}	
		}
	}
}


__global__ void Aloop_FW(float *d_a,int rowSize)
{

	int col =  blockIdx.x * blockDim.x + threadIdx.x;
	if(col >= rowSize)
		return;

	for(int k=0;k<rowSize;k++)
        {
           for(int i = 0; i < rowSize;i++)
           {
                for (int j=0;j< rowSize;j++)
                {

		d_a[i*rowSize+j] = fmin( d_a[i*rowSize+k] + d_a[k*rowSize+j] ,d_a[i*rowSize+j]);
		}
	    }
	}

}

void FW_A_loop(float* d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int size)
{

		Aloop_FW<<<1,1>>>(d_a,rowSize);
//		cudaThreadSynchronize();
}



void AFW(float* d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int d)
{
	int r = tilesize[d];
	if (r >= currSize)
		FW_A_loop(d_a,xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize);
	else
	{
		int newsize = currSize/r;
		for(int k=1; k<=r; k++) {
			AFW(d_a,(k-1)*newsize, (k-1)*newsize, (k-1)*newsize, (k-1)*newsize, (k-1)*newsize, (k-1)*newsize, newsize, d+1);
			
			for(int j=1; j<=r; j++) {
				if(j != k)
					BFW(d_a,(k-1)*newsize, (j-1)*newsize, (k-1)*newsize, (k-1)*newsize, (k-1)*newsize, (j-1)*newsize, newsize, d+1);
			}
			
			for(int i=1; i<=r; i++) {
				if(i != k)
					CFW(d_a,(i-1)*newsize, (k-1)*newsize, (i-1)*newsize, (k-1)*newsize, (k-1)*newsize, (k-1)*newsize, newsize, d+1);
			}
			
			for(int i=1; i<=r; i++) {
				for(int j=1; j<=r; j++) {
					if(i != k && j != k)
						DFW(d_a,(i-1)*newsize, (j-1)*newsize, (i-1)*newsize, (k-1)*newsize, (k-1)*newsize, (j-1)*newsize, newsize, d+1);
				}
			}	
		}
	}
}


int main(int argc, char *argv[])
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
        	//getchar();  
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
 

	err = cudaMemcpy(d_a, a, totalSize, cudaMemcpyHostToDevice);	

	struct timeval  tv1, tv2;
	gettimeofday(&tv1, NULL);

	AFW(d_a,0,0,0,0,0,0,rowSize,0);

	gettimeofday(&tv2, NULL);
	printf ("Total Execution time = %f seconds\n", (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec));

	err = cudaMemcpy(a, d_a, totalSize, cudaMemcpyDeviceToHost);

	print_matrix(a);

	return 0;
}

