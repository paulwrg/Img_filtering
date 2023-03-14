#!/bin/bash

make -j8

SRC=images/original/TimelyHugeGnu.gif
OUTPUT_DIR=images/processed
DEST=$OUTPUT_DIR/`basename $SRC .gif`-sobel.gif
echo "Running test on $SRC"

for i in {1..1}
do
    echo "-"
    mpirun -N 1 -n 1 ./sobelf $SRC $DEST
done