#!/bin/bash

# sshpass needs to be installed

for color in grey rgb; do
	for size in div4 div2 1 x2; do
		for processes in 1 4 9 16 25 36; do

			image="waterfall_${color}_${size}.raw"

			# Transfer image
			sshpass -p <CHANGE_ME> scp images/${image} std10128@linux22.di.uoa.gr:~/Parallel_Systems/MPI_OpenMP/

			# Execute time script
			sshpass -p <CHANGE_ME> ssh std10128@linux22.di.uoa.gr "cd Parallel_Systems/MPI_OpenMP/ ; ./time_MPI-OpenMP.sh ${color} ${size} ${processes}"

			# Delete image
			sshpass -p <CHANGE_ME> ssh std10128@linux22.di.uoa.gr "rm -f Parallel_Systems/MPI_OpenMP/${image}"

		done
	done
done
