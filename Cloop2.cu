#include<stdio.h>
#include<math.h>
#include <sys/time.h>

int rowSize;

__global__ void printGpu(float *d_a, int size)
{
	int i, j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
			printf("%0.1f\t", d_a[i * size + j]);
		printf("\n");
	}
}

__global__ void Cloop_FW(float *d_a, int k, int rowSize)
{
	__shared__ int intermed;
	int col =  blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= rowSize)
		return;
	for (int j = 0; j < rowSize; j++) {
		if (threadIdx.x == 0) {
			intermed = d_a[k * rowSize + j];
		}
		__syncthreads();

		d_a[col * rowSize + j ]  = fmin( d_a[col * rowSize + j ], d_a[col * rowSize + k] + intermed);
		__syncthreads();
	}
}

void print_matrix(float *d, int size)
{
	int i, j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
			printf("%0.1f\t", d[i * size + j]);
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
	int i, j, k;
	cudaError_t err = cudaSuccess;
	size_t totalSize = rowSize * colSize * sizeof(float);

	a = (float *) malloc(totalSize);
	if (!a)
	{
		printf("Unable to allocate memory for host array\n");
		return 1;
	}

	err = cudaMallocPitch(&d_a, &pitch, rowSize * sizeof(float), colSize);
	if (!d_a)
	{
		printf("memory failed for cudaMalloc");
		return 1;
	}

	if (err != 0) {
		printf("%s-%d", cudaGetErrorString(err), 3);
		return 1;
	}

	for (i = 0; i < rowSize; i++)
		for (j = 0; j < colSize; j++)
		{
			if (i == j) {
				a[i * rowSize + j] = 0;
			}
			else {

				a[i * rowSize + j] = (i + j) % 5 ? (i + j) : (i + j) % 7;
			}
		}

	//puts("input matrix :");
	//print_matrix(a,rowSize);

	err = cudaMemcpy(d_a, a, totalSize, cudaMemcpyHostToDevice);
	if (err != 0) {
		printf("after h2d %s-%d", cudaGetErrorString(err), 3);
		return 1;
	}
	int threadsPerBlock;
	//int noOfBlocks;

	if (rowSize < 1024)
	{
		threadsPerBlock	= rowSize;
	}
	else
	{
		threadsPerBlock = 1024;
	}

	//noOfBlocks = rowSize / threadsPerBlock;
	dim3 blocksPerGrid( (colSize + threadsPerBlock - 1) / threadsPerBlock , rowSize);
	struct timeval  tv1, tv2;
	gettimeofday(&tv1, NULL);

	for (k = 0; k < rowSize; k++) {
		Cloop_FW <<< blocksPerGrid, threadsPerBlock>>>(d_a, k, rowSize);
		cudaThreadSynchronize();
	}

	gettimeofday(&tv2, NULL);
	printf ("Total Execution time = %f seconds\n", (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec));

	printf("error = %s\n", cudaGetErrorString(cudaGetLastError()));

	err = cudaMemcpy(a, d_a, totalSize, cudaMemcpyDeviceToHost);
	if (err != 0) {
		printf("final %s-%d", cudaGetErrorString(err), 3);
		return 1;
	}

	//puts("output matrix :");
	//print_matrix(a,rowSize);

	free(a);
	cudaFree(d_a);
	return 0;
}
