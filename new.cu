#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<iostream>
#include<limits.h>
#include<algorithm>
#include<sys/time.h>
using namespace std;

#define INF           INT_MAX-1

int m;
int rowSize;
int tilesize[2] = {2, INT_MAX};

void print_matrix(float *d)
{
	int i,j;
	for(i=0;i<rowSize;i++)	
	{
		for(j=0;j<rowSize;j++)
			printf("%0.1f\t", d[i*rowSize+j]);
		puts("");
	}
}

__global__ void Dloop_FW(float *d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int rowSize)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + xColStart;
	if (j >= rowSize)
		return;
	//int i = xRowStart + rowSize*blockIdx.y;
	int i = blockIdx.y * blockDim.y + threadIdx.y + xRowStart;
	if (i >= rowSize)
        	return;

	__shared__ int intermed;
	for(int k = vRowStart; k < (vRowStart + currSize); k++) {
		if (threadIdx.x == 0) {
		       	intermed = d_a[i*rowSize + k];
	       	}

	        __syncthreads();
		if(i != j && j != k && i != k)
			d_a[i*rowSize + j]  = fmin(d_a[i*rowSize + j], intermed + d_a[k*rowSize+j]);
	}	
}

void FW_D_loop(float *d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int rowSize)
{        
	int threadsPerBlock;

	if (currSize <= 1024)
		threadsPerBlock = currSize;
	else
		threadsPerBlock = 1024;

	dim3 blocksPerGrid( currSize /threadsPerBlock ,currSize);

	Dloop_FW<<<blocksPerGrid,threadsPerBlock>>>(d_a, xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize, rowSize);
	//cudaThreadSynchronize();
}

void DFW(float *d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int d, int rowSize)
{
	int r = tilesize[d];
	if (r > currSize)
		FW_D_loop(d_a, xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize, rowSize);
	else
	{
		int newsize = currSize/r;
		for(int k=1; k<=r; k++) {	
			for(int i=1; i<=r; i++) {
				for(int j=1; j<=r; j++) {
					DFW(d_a, xRowStart + (i-1)*newsize, xColStart + (j-1)*newsize, uRowStart + (i-1)*newsize, uColStart + (k-1)*newsize, vRowStart + (k-1)*newsize, vColStart + (j-1)*newsize, newsize, d+1, rowSize);
				}
			}
			cudaThreadSynchronize();	
		}
	}
}

__global__ void Cloop_FW(float *d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int rowSize)
{
	__shared__ int intermed;
	int i =  blockIdx.x * blockDim.x + threadIdx.x + xRowStart;
	if(i >= rowSize)
		return;
	
	for(int k = vRowStart; k < (vRowStart + currSize); k++)
	{
		for(int j = xColStart; j < (xColStart + currSize); j++) 
		{
			if (threadIdx.x == 0) {
				intermed = d_a[k*rowSize+j];
			}
			__syncthreads();
			if(i != j && j != k && i != k)
				d_a[i*rowSize + j ]  = fmin( d_a[i*rowSize + j ], d_a[i*rowSize + k] + intermed);
	   	}	
	}
}

void FW_C_loop(float* d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int rowSize)
{        
	int threadsPerBlock;

        if (currSize <= 1024)
                threadsPerBlock = currSize;
        else
                threadsPerBlock = 1024;

        int noOfBlocks = currSize / threadsPerBlock;
	Cloop_FW<<<noOfBlocks,threadsPerBlock>>>(d_a, xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize, rowSize);

	//cudaThreadSynchronize();
}

void CFW(float* d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int d, int rowSize)
{
	int r = tilesize[d];
	if (r > currSize)
		FW_C_loop(d_a, xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize, rowSize);
	else
	{
		int newsize = currSize/r;
		for(int k=1; k<=r; k++) {	
			for(int i=1; i<=r; i++) {
				CFW(d_a, xRowStart + (i-1)*newsize, xColStart + (k-1)*newsize, uRowStart + (i-1)*newsize, uColStart + (k-1)*newsize, vRowStart + (k-1)*newsize, vColStart + (k-1)*newsize, newsize, d+1, rowSize);
			}
			cudaThreadSynchronize();

			for(int i=1; i<=r; i++) {
				for(int j=1; j<=r; j++) {
					if(j != k)
						DFW(d_a, xRowStart + (i-1)*newsize, xColStart + (j-1)*newsize, uRowStart + (i-1)*newsize, uColStart + (k-1)*newsize, vRowStart + (k-1)*newsize, vColStart + (j-1)*newsize, newsize, d+1, rowSize);
				}
			}
			cudaThreadSynchronize();	
		}
	}
}

__global__ void Bloop_FW(float *d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int rowSize)
{
	__shared__ int intermed;
	int j =  blockIdx.x * blockDim.x + threadIdx.x + xColStart;
	if(j >= rowSize)
		return;
	
	for(int k = vRowStart; k < (vRowStart + currSize); k++)
	{
		for(int i = xRowStart; i < (xRowStart + currSize); i++)
		{
			if (threadIdx.x == 0) {
				intermed = d_a[i*rowSize+k];
			}
			__syncthreads();
			if(i != j && j != k && i != k)
				d_a[i*rowSize +	j ]  = fmin(intermed + d_a[k*rowSize + j], d_a[i*rowSize+j]);
		}
	}
}

void FW_B_loop(float* d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int rowSize)
{        
	int threadsPerBlock;
	if (currSize < 1024)
        { 
                threadsPerBlock	= currSize;
       	}
	else
	{
		threadsPerBlock = 1024;
	}

	int noOfBlocks = currSize / threadsPerBlock;
	
	Bloop_FW<<<noOfBlocks,threadsPerBlock>>>(d_a, xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize, rowSize);
	//cudaThreadSynchronize();
}

void BFW(float* d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int d, int rowSize)
{
	int r = tilesize[d];
	if (r > currSize)
		FW_B_loop(d_a, xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize, rowSize);
	else
	{
		int newsize = currSize/r;
		for(int k=1; k<=r; k++) {
			for(int j=1; j<=r; j++) {
				BFW(d_a, xRowStart + (k-1)*newsize, xColStart + (j-1)*newsize, uRowStart + (k-1)*newsize, uColStart + (k-1)*newsize, vRowStart + (k-1)*newsize, vColStart + (j-1)*newsize, newsize, d+1, rowSize);
			}
			cudaThreadSynchronize();

			for(int i=1; i<=r; i++) {
				for(int j=1; j<=r; j++) {
					if(i != k)
						DFW(d_a, xRowStart + (i-1)*newsize, xColStart + (j-1)*newsize, uRowStart + (i-1)*newsize, uColStart + (k-1)*newsize, vRowStart + (k-1)*newsize, vColStart + (j-1)*newsize, newsize, d+1, rowSize);
				}
			}
			cudaThreadSynchronize();
		}
	}
}

__global__ void Aloop_FW(float *d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int rowSize)
{
	/*int col =  blockIdx.x * blockDim.x + threadIdx.x;
	if(col >= rowSize)
		return;
	*/
	for(int k = vRowStart; k < (vRowStart + currSize); k++)
	{
		for(int i = xRowStart; i < (xRowStart + currSize); i++)
		{
			for(int j = xColStart; j < (xColStart + currSize); j++) 
			{
				if(i != j && j != k && i != k)
					d_a[i*rowSize+j] = fmin( d_a[i*rowSize+k] + d_a[k*rowSize+j] ,d_a[i*rowSize+j]);
		}
	    }
	}
}

void FW_A_loop(float* d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int rowSize)
{        
	Aloop_FW<<<1,1>>>(d_a, xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize, rowSize);
}

void AFW(float* d_a, int xRowStart, int xColStart, int uRowStart, int uColStart, int vRowStart, int vColStart, int currSize, int d, int rowSize)
{
	int r = tilesize[d];
	if (r > currSize)
		FW_A_loop(d_a, xRowStart, xColStart, uRowStart, uColStart, vRowStart, vColStart, currSize, rowSize);
	else
	{
		int newsize = currSize/r;
		for(int k=1; k<=r; k++) {
			AFW(d_a, xRowStart + (k-1)*newsize, xColStart + (k-1)*newsize, uRowStart + (k-1)*newsize, uColStart + (k-1)*newsize, vRowStart + (k-1)*newsize, vColStart + (k-1)*newsize, newsize, d+1, rowSize);
			
			
			for(int j=1; j<=r; j++) {
				if(j != k)
					BFW(d_a, xRowStart + (k-1)*newsize, xColStart + (j-1)*newsize, uRowStart + (k-1)*newsize, uColStart + (k-1)*newsize, vRowStart + (k-1)*newsize, vColStart + (j-1)*newsize, newsize, d+1, rowSize);
			}
			
			for(int i=1; i<=r; i++) {
				if(i != k)
					CFW(d_a, xRowStart + (i-1)*newsize, xColStart + (k-1)*newsize, uRowStart + (i-1)*newsize, uColStart + (k-1)*newsize, vRowStart + (k-1)*newsize, vColStart + (k-1)*newsize, newsize, d+1, rowSize);
			}
			cudaThreadSynchronize();
			
			for(int i=1; i<=r; i++) {
				for(int j=1; j<=r; j++) {
					if(i != k && j != k)
						DFW(d_a, xRowStart + (i-1)*newsize, xColStart + (j-1)*newsize, uRowStart + (i-1)*newsize, uColStart + (k-1)*newsize, vRowStart + (k-1)*newsize, vColStart + (j-1)*newsize, newsize, d+1, rowSize);
				}
			}
			cudaThreadSynchronize();	
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
	
	err = cudaMemcpy(d_a, a, totalSize, cudaMemcpyHostToDevice);	

	struct timeval  tv1, tv2;
	gettimeofday(&tv1, NULL);

	AFW(d_a,0,0,0,0,0,0,rowSize,0, rowSize);

	gettimeofday(&tv2, NULL);
	printf ("Total Execution time = %f seconds\n", (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec));

	err = cudaMemcpy(a, d_a, totalSize, cudaMemcpyDeviceToHost);

	print_matrix(a);

	return 0;
}
