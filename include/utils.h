#include <stdint.h>
#include <mpi.h>

#define WORK_MODE_FAILURE  (0)
#define WORK_MODE_LEGACY   (1)
#define WORK_MODE_STRIPING (2)

MPI_Datatype kMPIPixelDatatype;

/* Represent one pixel from the image */
typedef struct pixel
{
    int r ; /* Red */
    int g ; /* Green */
    int b ; /* Blue */
} pixel ;

/* Represent one GIF image (animated or not */
typedef struct animated_gif
{
    int n_images ; /* Number of images */
    int * width ; /* Width of each image */
    int * height ; /* Height of each image */
    pixel ** p ; /* Pixels of each image */
    GifFileType * g ; /* Internal representation.
                         DO NOT MODIFY */
} animated_gif ;


extern void prepare_pixel_datatype(MPI_Datatype* datatype);