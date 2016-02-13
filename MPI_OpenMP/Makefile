all: 
	mpicc mpi+omp_convolute.c -std=c99 -lm -O2 -fopenmp

run:
	mpiexec -np 1 ./a.out -i ../images/waterfall_rgb.raw -o ../images/out.raw -w 1920 -h 2520 -bpp 3 -l 30
	
clean:
	rm a.out ../images/out.raw
