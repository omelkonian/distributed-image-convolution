#!/bin/bash

for color in grey rgb; do
	for size in div4 div2 1 x2 x4; do
		for blocks in 1 4 9 16; do 
			image="../images/waterfall_${color}_${size}.raw"		

			w1=1920
			h1=2520
			case $size in
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
			case $color in
				grey) 
					bpp=1;;
				rgb) 
					bpp=3;;
			esac			

			outputFile="CUDA_measurements.txt"

			time="$(./a.out -i ${image} -o out.raw -w ${w} -h ${h} -bpp ${bpp} -l 30 -b ${blocks})"						
			echo "${color}_${size}_${blocks}: ${time}" >> ${outputFile}						
		done
	done
done
