#!/bin/bash

make -j8

SRC=images/processed/giphy-3-sobel.gif
OUTPUT_DIR=images/processed
DEST=$OUTPUT_DIR/`basename $SRC .gif`-sobel.gif
echo "Running test on $SRC"

for i in {1..1}
do
    echo "-"
    mpirun -N 2 -n 2 ./sobelf $SRC $DEST
done