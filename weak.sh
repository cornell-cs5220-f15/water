#!/bin/bash

touch indices.txt

./submit wave 400 1 | cut -d '.' -f 1 >> indices.txt
./submit wave 800 4 | cut -d '.' -f 1 >> indices.txt
./submit wave 1100 8 | cut -d '.' -f 1 >> indices.txt
./submit wave 1400 12 | cut -d '.' -f 1 >> indices.txt