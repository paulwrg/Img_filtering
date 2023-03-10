/*
 * INF560
 *
 * Image Filtering Project
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include "gif_lib.h"
#include <mpi.h>
#include <omp.h>
#include "utils.h"
#include "filters.h"
#include "mpi_no_splitting.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0
#define WRITE_TO_FILE 1
#define filedebug 0

int sequential_main(int argc, char* argv[]) {
    char* input_filename;
    char* output_filename;
    animated_gif* image;
    struct timeval t1, t2;
    double duration;

    /* Check command-line arguments */
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels(input_filename);

    if (image == NULL) {
        return 1;
    }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);

    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
           input_filename, image->n_images, duration);

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    apply_all_filters(image);

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("SOBEL done in %lf s\n", duration);
    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);
    /* Store file from array of pixels to GIF file */
    if (!store_pixels(output_filename, image)) {
        return 1;
    }
    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("Export done in %lf s in file %s\n", duration, output_filename);

    return 0;
}

/*
 * Main entry point
 */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    prepare_pixel_datatype(&kMPIPixelDatatype);

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int mpi_world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    int ret_code;

    if (mpi_world_size == 1) {
        ret_code = sequential_main(argc, argv);
    } else if (mpi_rank == 0) {
        ret_code = master_main(argc, argv);
    } else {
        ret_code = slave_main(argc, argv);
    }

    MPI_Finalize();
    return ret_code;
}
