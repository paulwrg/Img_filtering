#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include <mpi.h>
#include <stddef.h>
#include "mpi_no_splitting.h"
#include "utils.h"
#include "filters.h"
#include "gif_lib.h"

/*
 * Old entry point
 */

int slave_main(int argc, char* argv[]) {
    int mpi_rank;
    int mpi_world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    animated_gif image;

    /* broadcast metadata */
    MPI_Bcast(&image.n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

    image.width = calloc(image.n_images, sizeof(int));
    image.height = calloc(image.n_images, sizeof(int));
    image.p = calloc(image.n_images, sizeof(pixel*));

    MPI_Bcast(image.width, image.n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image.height, image.n_images, MPI_INT, 0, MPI_COMM_WORLD);

    /* ask master for a new task */
    const int schedulingCommTag = image.n_images; // tag used for scheduling communications only
    int image_index = -1;
    MPI_Send(&image_index, 1, MPI_INT, 0, schedulingCommTag, MPI_COMM_WORLD);

    /* create array of all possible communication requests with master */
    MPI_Request processed_image_requests[image.n_images];
    for (int i = 0; i < image.n_images; ++i) {
        processed_image_requests[i] = MPI_REQUEST_NULL;
    }

    while (true) {
        /* check if there is another image to receive */
        MPI_Recv(&image_index, 1, MPI_INT, 0, schedulingCommTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (image_index == -1) {
            break;
        }

        /* Receive the image from master */
        image.p[image_index] = calloc(image.width[image_index] * image.height[image_index], sizeof(pixel));
        MPI_Recv(image.p[image_index], image.width[image_index] * image.height[image_index], kMPIPixelDatatype, 0,
                 image_index, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Convert the pixels into grayscale */
        apply_gray_filter(&image, image_index);

        /* Apply blur filter with convergence value */
        apply_blur_filter(&image, 5, 20, image_index);

        /* Apply sobel filter on pixels */
        apply_sobel_filter(&image, image_index);

        /* ask master for new task */
        MPI_Request req;
        MPI_Isend(&image_index, 1, MPI_INT, 0, schedulingCommTag, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);

        /* send processed image back to master */
        MPI_Isend(image.p[image_index], image.width[image_index] * image.height[image_index], kMPIPixelDatatype, 0,
                  image_index, MPI_COMM_WORLD, processed_image_requests + image_index);
    }

    /* check that all images have been safely sent to master process (requests resolved) */
    for (int i = 0; i < image.n_images; ++i) {
        if (processed_image_requests[i] != MPI_REQUEST_NULL) {
            MPI_Wait(processed_image_requests + i, MPI_STATUS_IGNORE);
        }
        free(image.p[i]);
    }

    free(image.width);
    free(image.height);
    free(image.p);

    return 0;
}

void do_master_work(animated_gif* image) {
    int mpi_rank;
    int mpi_world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    const int schedulingCommTag = image->n_images; // tag used for scheduling communications only

    /* broadcast metadata */
    MPI_Bcast(&image->n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image->width, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image->height, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);

    /* start scheduling */
    MPI_Request table_of_requests[mpi_world_size]; // communication requests with slaves
    int slave_signals[mpi_world_size];
    table_of_requests[0] = MPI_REQUEST_NULL; // no communication between master and self

    int n_processed_images = 0;
    int n_sent_images = 0;
    /* initialize communication request for each image in gif */
    MPI_Request processed_image_requests[image->n_images];
    for (int i = 0; i < image->n_images; ++i) {
        processed_image_requests[i] = MPI_REQUEST_NULL;
    }

    /* initialize communication with slaves */
    for (int slave_rank = 1; slave_rank < mpi_world_size; ++slave_rank) {
        MPI_Irecv(slave_signals + slave_rank, 1, MPI_INT, slave_rank, schedulingCommTag, MPI_COMM_WORLD, table_of_requests + slave_rank);
    }

    while (n_processed_images < image->n_images) {
        /* Wait until one request is completed */
        int slave_rank;
        MPI_Waitany(mpi_world_size, table_of_requests, &slave_rank, MPI_STATUS_IGNORE);

        int image_index = slave_signals[slave_rank];
        if (image_index != -1) {
            MPI_Irecv(image->p[image_index], image->width[image_index] * image->height[image_index],
                      kMPIPixelDatatype, slave_rank, image_index, MPI_COMM_WORLD, &processed_image_requests[image_index]);
            ++n_processed_images;
        }

        if (n_sent_images == image->n_images) {
            /* tell slave there is no work left to do, that it should exit while loop */
            MPI_Request req;
            int next_image_index = -1;
            MPI_Isend(&next_image_index, 1, MPI_INT, slave_rank, schedulingCommTag, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);
        } else {
            /* send next image to slave that just finished its work */
            MPI_Request req;
            int next_image_index = n_sent_images++;
            MPI_Isend(&next_image_index, 1, MPI_INT, slave_rank, schedulingCommTag, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);
            MPI_Isend(image->p[next_image_index], image->width[next_image_index] * image->height[next_image_index],
                      kMPIPixelDatatype, slave_rank, next_image_index, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req); // safe because only time we write here is when slave sends result

            /* wait for slave to confirm that work is done on this image */
            MPI_Irecv(slave_signals + slave_rank, 1, MPI_INT, slave_rank, schedulingCommTag, MPI_COMM_WORLD, table_of_requests + slave_rank);
        }
    }

    /* wait until all images have been correctly received */
    for (int i = 0; i < image->n_images; ++i) {
        if (processed_image_requests[i] != MPI_REQUEST_NULL) {
            MPI_Wait(processed_image_requests + i, MPI_STATUS_IGNORE);
        }
    }
}

int master_main(int argc, char* argv[]) {
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

    do_master_work(image);

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