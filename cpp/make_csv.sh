#!/bin/bash

if [ -z $1 ]; then
	echo "Error: No directory provided"
	exit	
fi

cd $1
touch timing.csv
namepattern='run([0-9]+).out'
for file in `ls -v *.out`; do
	if [[ $file =~ $namepattern ]]; then
		gridsize=${BASH_REMATCH[1]}
		frametime=`tail -n 1 $file`
		echo "$gridsize,$frametime" >> timing.csv
	fi
done

