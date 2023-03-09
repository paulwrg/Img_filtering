#include <mpi.h>
#include <stddef.h>
#include "utils.h"

void prepare_pixel_datatype(MPI_Datatype* datatype) {
    const int nitems = 3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[3];

    offsets[0] = offsetof(pixel, r);
    offsets[1] = offsetof(pixel, g);
    offsets[2] = offsetof(pixel, b);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, datatype);
    MPI_Type_commit(datatype);
}
