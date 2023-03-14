#!/bin/bash

make -j8

INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir $OUTPUT_DIR 2>/dev/null

for i in $INPUT_DIR/*gif ; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel.gif
    echo ""
    echo ""
    echo "Running test on $i -> $DEST"

    echo "---- SEQUENTIAL ----"
    mpirun -N 1 -n 1 ./sobelf_mpi_omp $i $DEST
    echo "--------------------"
    echo "---- MPI + OMP -----"
    mpirun -N 8 -n 8 ./sobelf_mpi_omp $i $DEST
    echo "--------------------"
    echo "------- CUDA -------"
    ./sobelf_cuda $i $DEST
    echo "--------------------"
done
