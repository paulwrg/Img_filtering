#include <mpi.h>
MPI_Datatype kMPIPixelDatatype;

#define CONV(l, c, nb_c) \
    ((l)*(nb_c)+(c))
    
extern void prepare_pixel_datatype(MPI_Datatype* datatype);