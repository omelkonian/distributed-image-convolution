#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

// #define PRINT

extern "C" void initiate(int height, int width, int blockSize, int matrixSize, unsigned char *input, unsigned char *output, int loops, int sum, int bpp, int *filterCPU);

/* Filters */					 
static int filters[4][3][3]={	
								{	{ 0, 0, 0},		// 0: identity
									{ 0, 1, 0},
									{ 0, 0, 0}	},

								{ 	{ 1, 0, 0},		// 1: blur
									{ 0, 1, 0},
									{ 0, 0, 1} 	}, 

								{ 	{-1,-1,-1},		// 2: edge detection
									{-1, 8,-1},
									{-1,-1,-1} 	},

								{ 	{-1,-1,-1},		// 3: sharpen
									{-1, 9,-1},
									{-1,-1,-1} 	}																
							}; 

#define filter filters[1] // Select which filter to be used here.

/* Arguments */
typedef struct {
	int width, height, bpp, loops;
	char inputImage[200], outputImage[200];
	int blockSize; // GPU block will be <blockSize> x <blockSize>
} Arguments;						

/* Utility functions */
void setArguments(char **argv, Arguments *args);
void printArguments(Arguments *args);

/* Main CUDA Program */
int main(int argc, char **argv) {
	/* Declarations */
	Arguments args;
	unsigned char *input, *output, *image, *buffer;	

	/* Check and set arguments */
	if (argc != 15) {
		printf("Usage: ./a.out -i <inputImg> -o <outputImg> -w <ImgWidth> -h <ImgHeight> -bpp <bitsPerPixel> -l <#loops> -b <blockSize>\n");
		exit(1);
	}
	else
		setArguments(argv, &args);
#ifdef PRINT
	printArguments(&args);
#endif

	/* Filter in CPU */
	int *filterCPU = (int *) malloc(9 * sizeof(int));
	for (int i = 0; i < 3; i++)
		memcpy(filterCPU + i*3, filter[i], 3 * sizeof(int));	
	
	/* Allocate CPU matrices (input, output) */
	int matrixSize = args.height * args.width * args.bpp * sizeof(unsigned char);
	input = (unsigned char *) malloc(matrixSize);
	output = (unsigned char *) malloc(matrixSize);

	/* Read input image */
	FILE *fp;
	fp = fopen(args.inputImage, "rb");
	fread(input, sizeof(unsigned char),args.height * args.width * args.bpp , fp);
	fclose(fp);		

	/* Calculate filter's sum */
	int sum = 0;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			sum += filter[i][j];
	sum = (sum != 0) ? sum : 1;

	/* Initiate GPU computation */
	initiate(args.height, args.width, args.blockSize, matrixSize, input, output, args.loops, sum, args.bpp, filterCPU);

	/* Write output to outputImage */
	fp = fopen(args.outputImage,"wb");
	fwrite(output, sizeof(unsigned char), args.height * args.width * args.bpp, fp);
	fclose(fp);

	/* Free resources */
	free(input);
	free(output);

	/* Terminate */
	exit(0);
}

/* Parses command-line arguments */
void setArguments(char **argv, Arguments *args) {
	for (int i = 1; i <= 14; i += 2) {
		if (strcmp(argv[i], "-i") == 0) 
			strcpy(args->inputImage, argv[i + 1]);
		else if (strcmp(argv[i], "-o") == 0) 
			strcpy(args->outputImage, argv[i + 1]);
		else if (strcmp(argv[i], "-w") == 0) 
			args->width = atoi(argv[i + 1]);
		else if (strcmp(argv[i], "-h") == 0) 
			args->height = atoi(argv[i + 1]);
		else if (strcmp(argv[i], "-bpp") == 0) 
			args->bpp = atoi(argv[i + 1]);
		else if (strcmp(argv[i], "-l") == 0) 
			args->loops = atoi(argv[i + 1]);
		else if (strcmp(argv[i], "-b") == 0) 
			args->blockSize = atoi(argv[i + 1]);
		else {
			printf("Usage: ./a.out -i <inputImg> -o <outputImg> -w <ImgWidth> -h <ImgHeight> -bpp <bitsPerPixel> -l <#loops> -b <blockSize>\n");
			exit(1);
		}
	}
}

/* Prints the command-line arguments to the console */
void printArguments(Arguments *args) {
	printf("Input: %s\nOutput: %s\n", args->inputImage, args->outputImage);
	printf("Width: %d Height: %d\nBpp: %d\n", args->width, args->height, args->bpp);
	printf("Loops: %d\n", args->loops);
	printf("Block Size: %d\n", args->blockSize);
}