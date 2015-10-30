#!/bin/bash

touch indices.txt

./submit wave 1000 1 | cut -d '.' -f 1 >> indices.txt
./submit wave 1000 4 | cut -d '.' -f 1 >> indices.txt
./submit wave 1000 8 | cut -d '.' -f 1 >> indices.txt
./submit wave 1000 10 | cut -d '.' -f 1 >> indices.txt
./submit wave 1000 20 | cut -d '.' -f 1 >> indices.txt
./submit wave 1000 40 | cut -d '.' -f 1 >> indices.txt
# ./submit wave 1000 40 | cut -d '.' -f 1 >> indices.txt