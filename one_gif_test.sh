#!/bin/bash

make -j8

echo "Running test on images/original/1.gif -> images/processed/1-sobel.gif"

mpirun -N 8 -n 8 ./sobelf images/original/1.gif images/processed/1-sobel.gif
