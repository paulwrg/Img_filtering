# Image Filtering
## Parallel and distributed algorithms

This project was done in the context of the INF560 course given in winter 2023 by M. Patrick Carribault at Ecole Polytechnique, and aimed at improving the performance of a provided sequential code by exploiting parallelism.

We used two different and separate methods:
- the first is a combination of MPI and OpenMP, using a master/slave model
- the second takes advantage of the computational power of the GPU through CUDA

The executable sobelf_cuda uses CUDA, while sobelf_mpi_omp either runs the sequential code if given parameters -N 1 -n 1, or uses the MPI + OpenMP paradigm for n > 2. The shell run_all_tests_one.sh evaluates all three techniques on a single gif, while run_all_tests_on_all_gifs.sh does so on all the gifs in the images/original directory. This allows for a comparison of their efficiency depending on the shape of the input gif (one vs many images, low vs high resolution).