#!/bin/bash

touch indices.txt

./submit wave 400 1 | cut -d '.' -f 1 >> indices.txt
./submit wave 640 4 | cut -d '.' -f 1 >> indices.txt
./submit wave 800 8 | cut -d '.' -f 1 >> indices.txt
./submit wave 920 12 | cut -d '.' -f 1 >> indices.txt