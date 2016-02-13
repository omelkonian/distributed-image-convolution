all: 
	nvcc -c gpu_convolute.cu
	nvcc convolute.cpp gpu_convolute.o

run:
	./a.out -i ../images/waterfall_grey_1.raw -o out.raw -w 1920 -h 2520 -bpp 1 -l 30 -b 16

clean:
	rm a.out *.o out.raw



