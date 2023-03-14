#!/bin/bash

make -j8
echo ""
SRC=images/original/Campusplan-Mobilitaetsbeschraenkte.gif
OUTPUT_DIR=images/processed
DEST=$OUTPUT_DIR/`basename $SRC .gif`-sobel.gif
echo "Running test on $SRC"


echo "---- SEQUENTIAL ----"
mpirun -N 1 -n 1 ./sobelf_mpi_omp $SRC $DEST
echo "--------------------"
echo "---- MPI + OMP -----"
mpirun -N 8 -n 8 ./sobelf_mpi_omp $SRC $DEST
echo "--------------------"
echo "------- CUDA -------"
./sobelf_cuda $SRC $DEST
echo "--------------------"
