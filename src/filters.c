#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include <stdlib.h>
#include "gif_lib.h"
#include <mpi.h>
#include <omp.h>
#include "mpi_no_splitting.h"
#include "filters.h"

void apply_gray_filter( animated_gif * image , int image_index)
{
    int i, j ;
    pixel ** p ;

    p = image->p ;

    #pragma omp parallel for private(j) num_threads(THREAD_NUM)
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

void apply_blur_filter( animated_gif * image, int size, int threshold , int image_index)
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

        #pragma omp parallel num_threads(THREAD_NUM)
        {
            #pragma omp for private(k) collapse(2)
            for (j = 0; j < height - 1; j++) {
                for (k = 0; k < width - 1; k++) {
                    new[CONV(j, k, width)].r = p[image_index][CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[image_index][CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[image_index][CONV(j, k, width)].b;
                }
            }

            /* Apply blur on top part of image (10%) */
            #pragma omp for private(k) collapse(2)
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
            int upper_bound = height * 0.9 + size; // need to explicit the upper bound or else won't compile here
            #pragma omp for private(k) collapse(2)
            for (j = height / 10 - size; j < upper_bound; j++) {
                for (k = size; k < width - size; k++) {
                    new[CONV(j, k, width)].r = p[image_index][CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[image_index][CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[image_index][CONV(j, k, width)].b;
                }
            }

            /* Apply blur on the bottom part of the image (10%) */
            #pragma omp for private(k) collapse(2)
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

            #pragma omp for private(k) collapse(2)
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
        }
    } while (threshold > 0 && !end);

#if SOBELF_DEBUG
    printf( "BLUR: number of iterations for image %d\n", n_iter ) ;
#endif

    free(new);

}

void apply_sobel_filter( animated_gif * image , int image_index)
{
    int i, j, k ;
    int width, height ;

    pixel ** p ;

    p = image->p ;

    width = image->width[image_index];
    height = image->height[image_index];

    pixel* sobel;

    sobel = (pixel*)malloc(width * height * sizeof(pixel));

    #pragma omp parallel num_threads(THREAD_NUM)
    {
        #pragma omp for private(k) collapse(2)
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

        #pragma omp for collapse(2) private(k)
        for (j = 1; j < height - 1; j++) {
            for (k = 1; k < width - 1; k++) {
                p[image_index][CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
                p[image_index][CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
                p[image_index][CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
            }
        }
    }

    free(sobel);
}

void apply_all_filters(animated_gif* image)
{
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

/* ************************************************************
 *
 * With splitting
 *
 * ************************************************************
 */


void apply_gray_filter_with_splitting(animated_gif * image, int image_index, int start, int stop)
{
    int i, j ;
    pixel ** p ;

    int mpi_rank;
    int mpi_world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    printf("G1 from rank %d\n", mpi_rank);

    p = image->p ;
    printf("G2 from rank %d\n", mpi_rank);

    // #pragma omp parallel for private(j) num_threads(THREAD_NUM)
    for (j = (image->width[image_index]) * start; j < (image->width[image_index]) * stop; j++) {
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

void apply_blur_filter_with_splitting(animated_gif * image, int size, int threshold, int image_index, int start, int stop)
{
    int i, j, k ;
    int width, height ;
    int endloop = 1 ;
    int n_iter = 0 ;
    int begin, end ;

    /* debug */
    int mpi_rank;
    int mpi_world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    pixel ** p ;
    pixel * new ;

    /* Get the pixels of all images */
    p = image->p ;

    /* Process all images */
    // n_iter = 0;
    width = image->width[image_index];
    height = image->height[image_index];

    /* Allocate array of new pixels */
    new = (pixel*)malloc(width * height * sizeof(pixel));

    /* Perform at least one blur iteration */
    // do {
        // endloop = 1;
        // n_iter++;

        #pragma omp parallel private(begin) private(end) num_threads(THREAD_NUM)
        {
            begin = start;
            // printf("Blur1 from rank %d\n", mpi_rank);
            end = stop < height - 1 ? stop : height - 1;
            // #pragma omp for private(k) collapse(2)
            for (j = begin; j < end; j++) {
                // begin = start;
                // end = min(stop, width - 1);
                for (k = 0; k < width - 1; k++) {
                    new[CONV(j, k, width)].r = p[image_index][CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[image_index][CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[image_index][CONV(j, k, width)].b;
                }
            }
            // printf("Blur2 from rank %d\n", mpi_rank);

            /* Apply blur on top part of image (10%) */
            begin = size;
            end = stop < height / 10 - size ? stop : height / 10 - size;
            // printf("Blur3 from rank %d\n", mpi_rank);
            #pragma omp for private(k) collapse(2)
            for (j = begin; j < end; j++) {
            // for (j = size; j < height / 10 - size; j++) {
            //     begin = max(start, size);
            //     end = min(stop, width - size);
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
            // printf("Blur4 from rank %d\n", mpi_rank);

            /* Copy the middle part of the image */
            int upper_bound = height * 0.9 + size; // need to explicit the upper bound or else won't compile here

            begin = start > height / 10 - size ? start : height / 10 - size;
            end = stop < height * 0.9 + size ? stop : height * 0.9 + size;
            // printf("Blur5 from rank %d\n", mpi_rank);
            #pragma omp for private(k) collapse(2)
            for (j = begin; j < end; j++) {
            // for (j = height / 10 - size; j < upper_bound; j++) {
            //     begin = max(start, size);
            //     end = min(stop, width - size);
            //     for (k = begin; k < end; k++) {
                for (k = size; k < width - size; k++) {
                    new[CONV(j, k, width)].r = p[image_index][CONV(j, k, width)].r;
                    new[CONV(j, k, width)].g = p[image_index][CONV(j, k, width)].g;
                    new[CONV(j, k, width)].b = p[image_index][CONV(j, k, width)].b;
                }
            }
            // printf("Blur6 from rank %d\n", mpi_rank);

            /* Apply blur on the bottom part of the image (10%) */
            begin = start > height * 0.9 + size ? start : height * 0.9 + size;
            end = stop < height - size ? stop : height - size;
            // printf("Blur7 from rank %d\n", mpi_rank);
            #pragma omp for private(k) collapse(2)
            for (j = begin; j < end; j++) {
            // for (j = height * 0.9 + size; j < height - size; j++) {
            //     begin = max(start, size);
            //     end = min(stop, width - size);
            //     for (k = begin; k < end; k++) {
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
            // printf("Blur8 from rank %d\n", mpi_rank);

            begin = start > 1 ? start : 1;
            end = stop < height - 1 ? stop : height - 1;
            #pragma omp for private(k) collapse(2)
            for (j = begin; j < end; j++) {
            // for (j = 1; j < height - 1; j++) {
            //     begin = max(start, 1);
            //     end = min(stop, width - 1);
            //     for (k = begin; k < end; k++) {
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
                        endloop = 0;
                    }

                    p[image_index][CONV(j, k, width)].r = new[CONV(j, k, width)].r;
                    p[image_index][CONV(j, k, width)].g = new[CONV(j, k, width)].g;
                    p[image_index][CONV(j, k, width)].b = new[CONV(j, k, width)].b;
                }
            }
        }
    // } while (threshold > 0 && !endloop);

#if SOBELF_DEBUG
    printf( "BLUR: number of iterations for image %d\n", n_iter ) ;
#endif

    free(new);
    return endloop;
}

void apply_sobel_filter_with_splitting(animated_gif * image, int image_index, int start, int stop)
{
    int i, j, k ;
    int width, height ;
    int begin, end ;

    pixel ** p ;

    p = image->p ;

    width = image->width[image_index];
    height = image->height[image_index];

    pixel* sobel;

    sobel = (pixel*)malloc(width * height * sizeof(pixel));

    #pragma omp parallel num_threads(THREAD_NUM) private(begin) private(end) 
    {
        begin = max(start, 1);
        end = min(stop, height - 1);
        #pragma omp for private(k) collapse(2)
        for (j = begin; j < end; j++) {
        // for (j = 1; j < height - 1; j++) {
        //     begin = max(start, 1);
        //     end = min(stop, width - 1);
        //     for (k = begin; k < end; k++) {
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

        begin = max(start, 1);
        end = min(stop, height - 1);
        #pragma omp for collapse(2) private(k)
        for (j = begin; j < end; j++) {
        // for (j = 1; j < height - 1; j++) {
        //     begin = max(start, 1);
        //     end = min(stop, width - 1);
        //     for (k = begin; k < end; k++) {
            for (k = 1; k < width - 1; k++) {
                p[image_index][CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
                p[image_index][CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
                p[image_index][CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
            }
        }
    }

    free(sobel);
}