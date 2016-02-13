#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/denominator)

/* Filter on GPU constant memory */
__constant__ int filterGPU[9];

/* Indexes a 2D array which is contiguously allocated in memory as a 1D array at position: (row, column) */
__device__ inline unsigned char* indexAt(unsigned char *array, int width, int bpp, int row, int col) {
	return &array[(row * width + col) * bpp];
}

__device__ inline int* indexAt2(int *array, int width, int bpp, int row, int col) {
	return &array[(row * width + col) * bpp];
}

/* Convolution on GPU (1 thread per pixel) */
__global__ void convolute(unsigned char *image, unsigned char *buffer, int sum, int height, int width, int bpp) { 	
	// Get thread's coordinates
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	int Y = threadIdx.y + blockIdx.y * blockDim.y;	

	// If it is inner pixel, convolute
	if ((X > 0) && (Y > 0) && (X < height - 1) && (Y < width - 1)) { 
		for (int offset = 0; offset < bpp; offset++) {			
			int newValue = 	(*(indexAt(image, width, bpp, X - 1	, Y - 1	) + offset)) * (*indexAt2(filterGPU, 3, 1, 0, 0))
						  +	(*(indexAt(image, width, bpp, X - 1	, Y    	) + offset)) * (*indexAt2(filterGPU, 3, 1, 0, 1))
						  + (*(indexAt(image, width, bpp, X - 1	, Y + 1	) + offset)) * (*indexAt2(filterGPU, 3, 1, 0, 2))
						  + (*(indexAt(image, width, bpp, X    	, Y - 1	) + offset)) * (*indexAt2(filterGPU, 3, 1, 1, 0))
						  + (*(indexAt(image, width, bpp, X  	, Y    	) + offset)) * (*indexAt2(filterGPU, 3, 1, 1, 1))
						  + (*(indexAt(image, width, bpp, X    	, Y + 1	) + offset)) * (*indexAt2(filterGPU, 3, 1, 1, 2))
						  + (*(indexAt(image, width, bpp, X + 1	, Y - 1	) + offset)) * (*indexAt2(filterGPU, 3, 1, 2, 0))
						  + (*(indexAt(image, width, bpp, X + 1	, Y    	) + offset)) * (*indexAt2(filterGPU, 3, 1, 2, 1))
						  + (*(indexAt(image, width, bpp, X + 1	, Y + 1	) + offset)) * (*indexAt2(filterGPU, 3, 1, 2, 2));

			newValue /= sum;

			if (newValue > UCHAR_MAX) newValue = UCHAR_MAX;
			else if (newValue < 0) newValue = 0;

			*indexAt(buffer, width, bpp, X, Y) = newValue;
		}
	}
	// Otherwise (if valid coordinates), just copy pixel
	else if ((X >= 0) && (X < height) && (Y >= 0) && (Y < width))
		*indexAt(buffer, width, bpp, X, Y) = *indexAt(image, width, bpp, X, Y);	
}

extern "C" void initiate(int height, int width, int blockSize, int matrixSize, unsigned char *input, unsigned char *output, int loops, int sum, int bpp, int *filterCPU){
	/* Declarations */
	unsigned char *image, *buffer;
	cudaEvent_t start, stop;
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));

	/* Allocate GPU matrices (image, buffer) */
	CUDA_SAFE_CALL(cudaMalloc((void**) &image, matrixSize));
	CUDA_SAFE_CALL(cudaMalloc((void**) &buffer, matrixSize));

	/* Calculate grid size */
	int gridX = FRACTION_CEILING(height, blockSize);
	int gridY = FRACTION_CEILING(width, blockSize);

	dim3 block(blockSize, blockSize);
	dim3 grid(gridX, gridY);

	/* Copy filter from CPU to GPU (constant memory) */
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(filterGPU, filterCPU, 9 * sizeof(int)));

	/* Initialize image */
	CUDA_SAFE_CALL(cudaMemcpy(image, input, matrixSize, cudaMemcpyHostToDevice));	


	/* Start time */
	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	/* Main Loop (No convergence check) */
	for (int loop = 0; loop < loops; loop++) {
		// Convolution on GPU
		convolute<<<grid, block>>>(image, buffer, sum, height, width, bpp);
		CUDA_SAFE_CALL(cudaGetLastError());
		
		// Synchronize threads
		CUDA_SAFE_CALL(cudaThreadSynchronize());

		// Swap buffers
		unsigned char *temp = image;
		image = buffer;
		buffer = temp;
	}

	/* Stop time */
	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));

	float elapsedTime;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("%3.1f ms\n", elapsedTime);
	

	/* Copy GPU's image to CPU's output */
	CUDA_SAFE_CALL(cudaMemcpy(output, image, matrixSize, cudaMemcpyDeviceToHost));

	/* Free resources */
	CUDA_SAFE_CALL(cudaFree(image));
	CUDA_SAFE_CALL(cudaFree(buffer));

	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(stop));
}
