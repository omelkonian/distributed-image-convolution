#!/bin/bash

#Usage: ./time_MPI.sh <color> <size> <processes>

color=$1
size=$2
processes=$3
w1=1920
h1=2520
case size in
	div4) 
		let w=w1/4
		let h=h1/4;;
	div2) 
		let w=w1/2
		let h=h1/2;;
	1) 
		let w=w1
		let h=h1;;
	x2)
		let w=w1*2
		let h=h1*2;;										
	x4) 
		let w=w1*4
		let h=h1*4;;
	x8) 
		let w=w1*8
		let h=h1*8;;															
esac
case color in
	grey) 
		bpp=1;;
	rgb) 
		bpp=3;;
esac			

outputFile="MPI_measurements.txt"
image="waterfall_${color}_${size}.raw"

time="$(mpiexec -f machines -np "${processes}" ./a.out -i "${image}" -o out.raw -w "$w" -h "$h" -bpp "${bpp}" -l 30)"						
echo "${color}_${size}_${processes}: ${time}" >> ${outputFile}			
