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
#include "filter_cuda.h"
#include "utilmpi.h"
#include <omp.h>

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0
#define WRITE_TO_FILE 1
#define filedebug 0
#define USE_CUDA 0
MPI_Datatype kMPIPixelDatatype;

void
apply_gray_filter( animated_gif * image , int image_index)
{
    int i, j ;
    pixel ** p ;

    p = image->p ;

    #pragma omp parallel
    for (j = 0; j < image->width[image_index] * image->height[image_index]; j++) {
        int moy;
        moy = (p[image_index][j].r + p[image_index][j].g + p[image_index][j].b) / 3;
        if (moy < 0) {
            moy = 0;
        }
        if (moy > 255) {
            moy = 255;
        }
        p[image_index][j].r = moy;
        p[image_index][j].g = moy;
        p[image_index][j].b = moy;
    }
}


void
apply_blur_filter( animated_gif * image, int size, int threshold , int image_index)
{
    int i, j, k ;
    int width, height ;
    int end = 0 ;
    int n_iter = 0 ;

    pixel ** p ;
    pixel * new ;

    /* Get the pixels of all images */
    p = image->p ;


    /* Process all images */
    n_iter = 0;
    width = image->width[image_index];
    height = image->height[image_index];

    /* Allocate array of new pixels */
    new = (pixel*)malloc(width * height * sizeof(pixel));


    /* Perform at least one blur iteration */
    do {
        end = 1;
        n_iter++;


        for (j = 0; j < height - 1; j++) {
            for (k = 0; k < width - 1; k++) {
                new[CONV(j, k, width)].r = p[image_index][CONV(j, k, width)].r;
                new[CONV(j, k, width)].g = p[image_index][CONV(j, k, width)].g;
                new[CONV(j, k, width)].b = p[image_index][CONV(j, k, width)].b;
            }
        }

        /* Apply blur on top part of image (10%) */
        for (j = size; j < height / 10 - size; j++) {
            for (k = size; k < width - size; k++) {
                int stencil_j, stencil_k;
                int t_r = 0;
                int t_g = 0;
                int t_b = 0;

                for (stencil_j = -size; stencil_j <= size; stencil_j++) {
                    for (stencil_k = -size; stencil_k <= size; stencil_k++) {
                        t_r += p[image_index][CONV(j + stencil_j, k + stencil_k, width)].r;
                        t_g += p[image_index][CONV(j + stencil_j, k + stencil_k, width)].g;
                        t_b += p[image_index][CONV(j + stencil_j, k + stencil_k, width)].b;
                    }
                }

                new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
            }
        }

        /* Copy the middle part of the image */
        for (j = height / 10 - size; j < height * 0.9 + size; j++) {
            for (k = size; k < width - size; k++) {
                new[CONV(j, k, width)].r = p[image_index][CONV(j, k, width)].r;
                new[CONV(j, k, width)].g = p[image_index][CONV(j, k, width)].g;
                new[CONV(j, k, width)].b = p[image_index][CONV(j, k, width)].b;
            }
        }

        /* Apply blur on the bottom part of the image (10%) */
        for (j = height * 0.9 + size; j < height - size; j++) {
            for (k = size; k < width - size; k++) {
                int stencil_j, stencil_k;
                int t_r = 0;
                int t_g = 0;
                int t_b = 0;

                for (stencil_j = -size; stencil_j <= size; stencil_j++) {
                    for (stencil_k = -size; stencil_k <= size; stencil_k++) {
                        t_r += p[image_index][CONV(j + stencil_j, k + stencil_k, width)].r;
                        t_g += p[image_index][CONV(j + stencil_j, k + stencil_k, width)].g;
                        t_b += p[image_index][CONV(j + stencil_j, k + stencil_k, width)].b;
                    }
                }

                new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
            }
        }

        for (j = 1; j < height - 1; j++) {
            for (k = 1; k < width - 1; k++) {

                float diff_r;
                float diff_g;
                float diff_b;

                diff_r = (new[CONV(j, k, width)].r - p[image_index][CONV(j, k, width)].r);
                diff_g = (new[CONV(j, k, width)].g - p[image_index][CONV(j, k, width)].g);
                diff_b = (new[CONV(j, k, width)].b - p[image_index][CONV(j, k, width)].b);

                if (diff_r > threshold || -diff_r > threshold
                    ||
                    diff_g > threshold || -diff_g > threshold
                    ||
                    diff_b > threshold || -diff_b > threshold
                        ) {
                    end = 0;
                }

                p[image_index][CONV(j, k, width)].r = new[CONV(j, k, width)].r;
                p[image_index][CONV(j, k, width)].g = new[CONV(j, k, width)].g;
                p[image_index][CONV(j, k, width)].b = new[CONV(j, k, width)].b;
            }
        }

    } while (threshold > 0 && !end);

#if SOBELF_DEBUG
    printf( "BLUR: number of iterations for image %d\n", n_iter ) ;
#endif

    free(new);

}

void
apply_sobel_filter( animated_gif * image , int image_index)
{
    int i, j, k ;
    int width, height ;

    pixel ** p ;

    p = image->p ;

    width = image->width[image_index];
    height = image->height[image_index];

    pixel* sobel;

    sobel = (pixel*)malloc(width * height * sizeof(pixel));

    for (j = 1; j < height - 1; j++) {
        for (k = 1; k < width - 1; k++) {
            int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
            int pixel_blue_so, pixel_blue_s, pixel_blue_se;
            int pixel_blue_o, pixel_blue, pixel_blue_e;

            float deltaX_blue;
            float deltaY_blue;
            float val_blue;

            pixel_blue_no = p[image_index][CONV(j - 1, k - 1, width)].b;
            pixel_blue_n = p[image_index][CONV(j - 1, k, width)].b;
            pixel_blue_ne = p[image_index][CONV(j - 1, k + 1, width)].b;
            pixel_blue_so = p[image_index][CONV(j + 1, k - 1, width)].b;
            pixel_blue_s = p[image_index][CONV(j + 1, k, width)].b;
            pixel_blue_se = p[image_index][CONV(j + 1, k + 1, width)].b;
            pixel_blue_o = p[image_index][CONV(j, k - 1, width)].b;
            pixel_blue = p[image_index][CONV(j, k, width)].b;
            pixel_blue_e = p[image_index][CONV(j, k + 1, width)].b;

            deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 2 * pixel_blue_e - pixel_blue_so +
                          pixel_blue_se;

            deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2 * pixel_blue_n -
                          pixel_blue_no;

            val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;


            if (val_blue > 50) {
                sobel[CONV(j, k, width)].r = 255;
                sobel[CONV(j, k, width)].g = 255;
                sobel[CONV(j, k, width)].b = 255;
            } else {
                sobel[CONV(j, k, width)].r = 0;
                sobel[CONV(j, k, width)].g = 0;
                sobel[CONV(j, k, width)].b = 0;
            }
        }
    }

    for (j = 1; j < height - 1; j++) {
        for (k = 1; k < width - 1; k++) {
            p[image_index][CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
            p[image_index][CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
            p[image_index][CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
        }
    }

    free(sobel);


}

void apply_all_filters(animated_gif* image) {
    for (int i = 0; i < image->n_images; ++i) {
        // Convert the pixels into grayscale
        apply_gray_filter(image, i);
    }

    for (int i = 0; i < image->n_images; ++i) {
        // Apply blur filter with convergence value
        apply_blur_filter(image, 5, 20, i);
    }

    for (int i = 0; i < image->n_images; ++i) {
        // Apply sobel filter on pixels
        apply_sobel_filter(image, i);
    }
}

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

#if USE_CUDA
        apply_all_filters_gpu(&image);
#else
        apply_all_filters(&image);
#endif

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

#if USE_CUDA
    apply_all_filters_gpu(image);
    printf("***using cuda***\n");
#else
    apply_all_filters(image);
#endif

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
