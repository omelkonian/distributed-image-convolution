#!/bin/bash

# sshpass needs to be installed

for color in grey rgb; do
	for size in div4 div2 1 x2; do
		image="waterfall_${color}_${size}.raw"

		# Transfer image
		sshpass -p <CHANGE_ME> scp images/${image} std10128@linux22.di.uoa.gr:~/Parallel_Systems/MPI/
		echo "	Image transfered"

		for processes in 1 4 9 16 25 36; do
			# Execute time script
			sshpass -p <CHANGE_ME> ssh std10128@linux22.di.uoa.gr "cd Parallel_Systems/MPI/ ; ./time_MPI.sh ${color} ${size} ${processes}"
		done
		echo "	Executions finished"

		# Delete image
		sshpass -p <CHANGE_ME> ssh std10128@linux22.di.uoa.gr "rm -f Parallel_Systems/MPI/${image}"
		echo "	Image deleted"
	done
done
