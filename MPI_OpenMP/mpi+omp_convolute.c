#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "mpi.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define CHECK 10
#define MASTER if (rank == 0)
// #define PRINT

/* Filters */
static int sum = 0;							 
static int filters[7][3][3]={	
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
									{-1,-1,-1} 	},

								{ 	{ 0, 1, 0},		// 4: edge detection(2)
									{ 1,-4, 1},
									{ 0, 1, 0} 	},

								{ 	{-2,-1, 0},		// 5: emboss
									{-1, 1, 1},
									{ 0, 1, 2} 	},

								{ 	{ 0, 0, 0},		// 6: shift
									{ 0, 0, 0},
									{ 0, 1, 0} 	}																		
							}; 

#define filter filters[1] // Select which filter to be used here.

/* Arguments */
typedef struct {
	int width, height, bpp, loops;
	char inputImage[200], outputImage[200];
} Arguments;

/* Utility functions */
void setArguments(char **argv, Arguments *args);
void printArguments();
void printCartGrid(MPI_Comm comm, int comm_size);
unsigned char* indexAt(unsigned char *array, int width, int bpp, int row, int col);
void convolute(unsigned char *image, unsigned char *buffer, int row, int col, int width, int bpp);
void printOuter(unsigned char *image, int w, int bpp, int h);

/* Main MPI Program */
int main(int argc, char **argv) {
	/* Declarations */
	int rank, comm_size;
	Arguments args;

	/* Initialize MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

#ifdef _OPENMP
	MASTER printf("OPENMP is supported!\n");
#else
	MASTER printf("OPENMP is not supported. Aborting..."\n);
	MPI_Finalize();
	exit(1);
#endif

	/* Check and set arguments */
	if (argc != 13) {
		MASTER printf("Usage: ./a.out -i <inputImg> -o <outputImg> -w <ImgWidth> -h <ImgHeight> -bpp <bitsPerPixel> -l <#loops>");
		exit(1);
	}
	else
		setArguments(argv, &args);
	
#ifdef PRINT
	MASTER printArguments(&args);    
#endif

	int bpp = args.bpp;

	/* Cartesian topology setup */
	MPI_Comm old_comm, new_comm;
	old_comm = MPI_COMM_WORLD;
	int dim = sqrt(comm_size);
	int dim_size[2] = {dim, dim}, periods[2] = {1, 1}; /* Set periodicity on both dimentions */	
	MPI_Cart_create(old_comm, 2, dim_size, periods, 1, &new_comm); /* let MPI re-arrange processes for optimal processor assignment*/

	/* Create buffers */
	int subWidth = args.width/dim_size[1];
	int subHeight = args.height/dim_size[0];
		
	unsigned char *image  = malloc(bpp * (subWidth + 2) * (subHeight + 2) * sizeof(unsigned char));
	unsigned char *buffer = malloc(bpp * (subWidth + 2) * (subHeight + 2) * sizeof(unsigned char));

#ifdef PRINT
	MASTER printf("SubW: %d    SubH: %d\n", subWidth, subHeight);
	MASTER printCartGrid(new_comm, comm_size);
#endif

	/* Get neighbours needed later for new_communication */
	int me[2];
	MPI_Cart_coords(new_comm, rank, 2, me);    	
	int up=-1, down=-1, left=-1, right=-1, upleft=-1, upright=-1, downleft=-1, downright=-1; /* the eight neighbors*/
	
	MPI_Cart_shift(new_comm, 0, 1, &up, &down);    
    MPI_Cart_shift(new_comm, 1, 1, &left, &right);

	int coords[2] = {me[0] - 1, me[1] - 1};
	MPI_Cart_rank(new_comm, coords, &upleft);
	coords[0] = me[0] - 1; coords[1] = me[1] + 1;
	MPI_Cart_rank(new_comm, coords, &upright);
	coords[0] = me[0] + 1; coords[1] = me[1] - 1;
	MPI_Cart_rank(new_comm, coords, &downleft);
	coords[0] = me[0] + 1; coords[1] = me[1] + 1;
	MPI_Cart_rank(new_comm, coords, &downright);

#ifdef PRINT
	printf("[%d]: %d %d %d %d %d %d %d %d\n", rank, up, down, left, right, upleft, upright, downleft, downright);
#endif

	/* Derived Datatypes */		

	// ROW
	MPI_Datatype Row;
	MPI_Type_vector(subWidth, bpp, bpp, MPI_UNSIGNED_CHAR, &Row);
	MPI_Type_commit(&Row);
	
	// COLUMN
	MPI_Datatype Column;
	MPI_Type_vector(subHeight, bpp, bpp*(subWidth+2), MPI_UNSIGNED_CHAR, &Column);
	MPI_Type_commit(&Column);

	// INNER_ARRAY
	MPI_Datatype InnerArray;    
	int arrayOfSizes[2] = {args.height, args.width * bpp};
	int arrayOfSubsizes[2] = {subHeight, subWidth * bpp};
	int arrayOfStarts[2] = {me[0] * subHeight, me[1] * subWidth * bpp};

	MPI_Type_create_subarray(2, arrayOfSizes, arrayOfSubsizes, arrayOfStarts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &InnerArray);
	MPI_Type_commit(&InnerArray);

	/* Read from input image file */
	MPI_File file;
	MPI_File_open(new_comm, args.inputImage, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
	MPI_File_set_view(file, 0, MPI_UNSIGNED_CHAR, InnerArray, "native", MPI_INFO_NULL);
	MPI_File_read_all(file, image, arrayOfSubsizes[0] * arrayOfSubsizes[1], MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);    
	MPI_File_close(&file);

	/* Shift image: (0,0) -> (1,1) */
	for(int i = subHeight; i >= 0; i--)
		memmove(indexAt(image,subWidth+2,bpp,i+1,1), indexAt(image,subWidth,bpp,i,0), subWidth*bpp*sizeof(unsigned char));

	/* Persistent communication */
	MPI_Request recvReq[8][2], sendReq[8][2];		
	int w = subWidth, h = subHeight; // aliases 	

	// Up neighbor
	MPI_Recv_init(indexAt(image,w+2,bpp,0,1), 		1, 	Row, 	up, 	1, 		new_comm, &recvReq[0][0]);
	MPI_Send_init(indexAt(image,w+2,bpp,1,1),		1, 	Row, 	up, 	2, 		new_comm, &sendReq[0][0]);
	MPI_Recv_init(indexAt(buffer,w+2,bpp,0,1), 		1, 	Row, 	up, 	1, 		new_comm, &recvReq[0][1]);
	MPI_Send_init(indexAt(buffer,w+2,bpp,1,1),		1, 	Row, 	up, 	2, 		new_comm, &sendReq[0][1]);		
	
	// Down neighbor
	MPI_Recv_init(indexAt(image,w+2,bpp,h+1,1),		1, 	Row, 	down, 	2, 		new_comm, &recvReq[1][0]);
	MPI_Send_init(indexAt(image,w+2,bpp,h,1), 		1, 	Row, 	down, 	1, 		new_comm, &sendReq[1][0]);
	MPI_Recv_init(indexAt(buffer,w+2,bpp,h+1,1),	1, 	Row, 	down, 	2, 		new_comm, &recvReq[1][1]);
	MPI_Send_init(indexAt(buffer,w+2,bpp,h,1), 		1, 	Row, 	down, 	1, 		new_comm, &sendReq[1][1]);

	// Left neighbor
	MPI_Recv_init(indexAt(image,w+2,bpp,1,0),		1, 	Column, left, 	3, 		new_comm, &recvReq[2][0]);
	MPI_Send_init(indexAt(image,w+2,bpp,1,1),		1, 	Column, left, 	4, 		new_comm, &sendReq[2][0]);
	MPI_Recv_init(indexAt(buffer,w+2,bpp,1,0),		1, 	Column, left, 	3, 		new_comm, &recvReq[2][1]);
	MPI_Send_init(indexAt(buffer,w+2,bpp,1,1),		1, 	Column, left, 	4, 		new_comm, &sendReq[2][1]);

	
	// Right neighbor
	MPI_Recv_init(indexAt(image,w+2,bpp,1,w+1),		1, 	Column, right, 	4, 		new_comm, &recvReq[3][0]);
	MPI_Send_init(indexAt(image,w+2,bpp,1,w),		1, 	Column, right, 	3, 		new_comm, &sendReq[3][0]);
	MPI_Recv_init(indexAt(buffer,w+2,bpp,1,w+1),	1, 	Column, right, 	4, 		new_comm, &recvReq[3][1]);
	MPI_Send_init(indexAt(buffer,w+2,bpp,1,w),		1, 	Column, right, 	3, 		new_comm, &sendReq[3][1]);	

	// Upleft neighbor
	MPI_Recv_init(indexAt(image,w+2,bpp,0,0),		bpp, MPI_UNSIGNED_CHAR, upleft, 	5, 	new_comm, &recvReq[4][0]);
	MPI_Send_init(indexAt(image,w+2,bpp,1,1),		bpp, MPI_UNSIGNED_CHAR, upleft, 	6, 	new_comm, &sendReq[4][0]);
	MPI_Recv_init(indexAt(buffer,w+2,bpp,0,0),		bpp, MPI_UNSIGNED_CHAR, upleft, 	5, 	new_comm, &recvReq[4][1]);
	MPI_Send_init(indexAt(buffer,w+2,bpp,1,1),		bpp, MPI_UNSIGNED_CHAR, upleft, 	6, 	new_comm, &sendReq[4][1]);
		
	// Upright neighbor
	MPI_Recv_init(indexAt(image,w+2,bpp,0,w+1),		bpp, MPI_UNSIGNED_CHAR, upright, 	7, 	new_comm, &recvReq[5][0]);
	MPI_Send_init(indexAt(image,w+2,bpp,1,w), 		bpp, MPI_UNSIGNED_CHAR, upright, 	8, 	new_comm, &sendReq[5][0]);
	MPI_Recv_init(indexAt(buffer,w+2,bpp,0,w+1),	bpp, MPI_UNSIGNED_CHAR, upright, 	7, 	new_comm, &recvReq[5][1]);
	MPI_Send_init(indexAt(buffer,w+2,bpp,1,w), 		bpp, MPI_UNSIGNED_CHAR, upright, 	8, 	new_comm, &sendReq[5][1]);
	
	// Downleft neighbor
	MPI_Recv_init(indexAt(image,w+2,bpp,h+1,0),		bpp, MPI_UNSIGNED_CHAR, downleft, 	8, 	new_comm, &recvReq[6][0]);
	MPI_Send_init(indexAt(image,w+2,bpp,h,1), 		bpp, MPI_UNSIGNED_CHAR, downleft, 	7, 	new_comm, &sendReq[6][0]);
	MPI_Recv_init(indexAt(buffer,w+2,bpp,h+1,0),	bpp, MPI_UNSIGNED_CHAR, downleft, 	8, 	new_comm, &recvReq[6][1]);
	MPI_Send_init(indexAt(buffer,w+2,bpp,h,1), 		bpp, MPI_UNSIGNED_CHAR, downleft, 	7, 	new_comm, &sendReq[6][1]);

	// Downright neighbor	
	MPI_Recv_init(indexAt(image,w+2,bpp,h+1,w+1),	bpp, MPI_UNSIGNED_CHAR, downright, 	6, 	new_comm, &recvReq[7][0]);	
	MPI_Send_init(indexAt(image,w+2,bpp,h,w), 		bpp, MPI_UNSIGNED_CHAR, downright, 	5, 	new_comm, &sendReq[7][0]);
	MPI_Recv_init(indexAt(buffer,w+2,bpp,h+1,w+1),	bpp, MPI_UNSIGNED_CHAR, downright, 	6, 	new_comm, &recvReq[7][1]);	
	MPI_Send_init(indexAt(buffer,w+2,bpp,h,w), 		bpp, MPI_UNSIGNED_CHAR, downright, 	5, 	new_comm, &sendReq[7][1]);

	/* Calculate sum of filter */
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			sum += filter[i][j];
	
	/* Start Time */	
	double startTime = MPI_Wtime();

	/* Main Loop */
	int allFinished = 0;	
	for (int loop = 0; loop < args.loops; loop++) {
		int index = loop % 2;

		/* Receive & Send messages */
		for (int i = 0; i < 8; i++) {
			MPI_Start(&sendReq[i][index]);
			MPI_Start(&recvReq[i][index]);					
		}

		/* Convolute inner pixels */
		#pragma omp parallel for shared(buffer) collapse(2) schedule(static)    
		for (int i = 2; i < h; i++)
			for (int j = 2; j < w; j++)
				convolute(image, buffer, i, j, w+2, bpp);
		
		/* Wait on receive requests */
		for (int i = 0; i < 8; i++)
			MPI_Wait(&recvReq[i][index], MPI_STATUS_IGNORE);	

		/* Convolute outer pixels */
		// #pragma omp parallel for shared(buffer) schedule(static)    
		for (int j = 1; j <= w; j++) {			
			convolute(image, buffer, 1, j, w+2, bpp);
			convolute(image, buffer, h, j, w+2, bpp);
		}		
		// #pragma omp parallel for shared(buffer) schedule(static)    
		for (int i = 1; i <= h; i++) {
			convolute(image, buffer, i, 1, w+2, bpp);
			convolute(image, buffer, i, w, w+2, bpp);
		}

		/* Check if we reached fixpoint (only every CHECK loops) */
		if (loop % CHECK == 0) {
			int myFinished = 1;
			for (int i = 1; i < subHeight + 1; i++)
				for (int j = 1; j < subWidth + 1; j++)
					if (*indexAt(image,subWidth+2,bpp,i,j) != *indexAt(buffer,subWidth+2,bpp,i,j)) {
						myFinished = 0;
						break;
					}
			MPI_Allreduce(&myFinished, &allFinished, 1, MPI_INT, MPI_MIN, new_comm);
		}

		/* Wait on send requests */
		for (int i = 0; i < 8; i++)
			MPI_Wait(&sendReq[i][index], MPI_STATUS_IGNORE);	

		/* Swap buffers */
		unsigned char *temp = buffer;
		buffer = image;
		image = temp;

		/* Terminate if we reached fixpoint */
		if (allFinished)
			break;
	}

	/* Reduce to the longest execution */	
	double localTime = MPI_Wtime() - startTime, globalTime;	
	MPI_Reduce(&localTime, &globalTime, 1, MPI_DOUBLE, MPI_MAX, 0, new_comm);
	MASTER printf("\nTotal time: %.10f\n", globalTime);

	/* Un-shift image: (1, 1) <- (0, 0) */
	for(int i=0; i<subHeight; i++)
		memmove(indexAt(image,subWidth,bpp,i,0), indexAt(image,subWidth+2,bpp,i+1,1), subWidth*bpp*sizeof(unsigned char));

	/* Write final result to output image file*/	
	MPI_File_open(new_comm, args.outputImage, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
	MPI_File_set_view(file, 0, MPI_UNSIGNED_CHAR, InnerArray, "native", MPI_INFO_NULL);
	MPI_File_write_all(file, image, arrayOfSubsizes[0] * arrayOfSubsizes[1], MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE);    
	MPI_File_close(&file);

	/* Free resources */
	MPI_Comm_free(&new_comm);

	free(buffer);
	free(image);

	MPI_Type_free(&Row);
	MPI_Type_free(&Column);
	MPI_Type_free(&InnerArray);

	for (int i = 0; i < 8; i++) {
		MPI_Request_free(&recvReq[i][0]);
		MPI_Request_free(&sendReq[i][0]);
		MPI_Request_free(&recvReq[i][1]);
		MPI_Request_free(&sendReq[i][1]);
	}		

	/* Terminate */
	MPI_Finalize();
	exit(0);	
}

/* Calculates the convolution of two 2D matrices for given position: (row, column) */
void convolute(unsigned char *image, unsigned char *buffer, int row, int col, int width, int bpp) { 	
	
	// #pragma omp parallel for num_threads(bpp) shared(buffer) schedule(static)
	for (int offset = 0; offset < bpp; offset++) {			
		int newValue = 	(*(indexAt(image, width, bpp, row - 1, col - 1) + offset)) * filter[0][0]
					  +	(*(indexAt(image, width, bpp, row - 1, col    ) + offset)) * filter[0][1]
					  + (*(indexAt(image, width, bpp, row - 1, col + 1) + offset)) * filter[0][2]
					  + (*(indexAt(image, width, bpp, row    , col - 1) + offset)) * filter[1][0]
					  + (*(indexAt(image, width, bpp, row 	 , col 	  ) + offset)) * filter[1][1]
					  + (*(indexAt(image, width, bpp, row 	 , col + 1) + offset)) * filter[1][2]
					  + (*(indexAt(image, width, bpp, row + 1, col - 1) + offset)) * filter[2][0]
					  + (*(indexAt(image, width, bpp, row + 1, col    ) + offset)) * filter[2][1]
					  + (*(indexAt(image, width, bpp, row + 1, col + 1) + offset)) * filter[2][2];

		newValue /= (sum == 0 ? 1 : sum);

		if (newValue > UCHAR_MAX) newValue = UCHAR_MAX;
		else if (newValue < 0) newValue = 0;

		*(indexAt(buffer, width, bpp, row, col) + offset) = newValue;
	}
}

/* Indexes a 2D array which is contiguously allocated in memory as a 1D array at position: (row, column) */
unsigned char* indexAt(unsigned char *array, int width, int bpp, int row, int col) {
	return &array[(row * width + col) * bpp];
}

/* Parses command-line arguments */
void setArguments(char **argv, Arguments *args) {
	for (int i = 1; i <= 12; i += 2) {
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
		else {
			printf("Usage: ./a.out -i <inputImg> -o <outputImg> -w <ImgWidth> -h <ImgHeight> -bpp <bitsPerPixel>");
			exit(1);
		}
	}
}

/* Prints the command-line arguments to the console */
void printArguments(Arguments *args) {
	printf("Input: %s\nOutput: %s\n", args->inputImage, args->outputImage);
	printf("Width: %d Height: %d\nBpp: %d\n", args->width, args->height, args->bpp);
	printf("Loops: %d\n", args->loops);
}

/* Prints the process topology to the console */
void printCartGrid(MPI_Comm comm, int comm_size) {
	int coords[2];
	int curI = 0;
	printf("-----------------\n");
	for (int i = 0; i < comm_size; i++) {
		MPI_Cart_coords(comm, i, 2, coords);
		if (coords[0] > curI) {
			printf("|\n-----------------\n");
			curI++;
		}		
		printf("| %d ", i);
	}
	printf("|\n");
	printf("-----------------\n");
}