#!/bin/bash

make -j8

SRC=images/original/Campusplan-Mobilitaetsbeschraenkte.gif
OUTPUT_DIR=images/processed
DEST=$OUTPUT_DIR/`basename $SRC .gif`-sobel.gif
echo "Running test on $SRC"

for i in {1..1}
do
    echo "-"
    mpirun -N 8 -n 8 ./sobelf $SRC $DEST
done