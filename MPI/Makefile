all: 
	mpicc mpi_convolute.c -std=c99 -lm -O2

run:
	@mpiexec -f machines -np 16 ./a.out -i ../images/waterfall_rgb.raw -o ../images/out.raw -w 1920 -h 2520 -bpp 3 -l 20
	
clean:
	rm a.out ../images/out.raw
