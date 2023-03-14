#!/bin/bash

make -j8

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

for i in $INPUT_DIR/*gif ; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo "Running test on $i -> $DEST"
    echo "Number of MPI processes 1 and number of threads 1"
    
    ./sobelf $i $DEST
done
