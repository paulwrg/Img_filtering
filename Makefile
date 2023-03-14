SRC_DIR=src
HEADER_DIR=include
OBJ_DIR_CUDA=obj
OBJ_DIR_MPI_OMP=obj_mpi_omp

CC=mpicc
NVCC = nvcc
CFLAGS=-O3 -I$(HEADER_DIR) -std=gnu99 
NVCFLAGS=-O3 -I$(HEADER_DIR) 
LDFLAGS=-lm -lmpi -lgomp  -L/usr/local/cuda/lib64 -lcudart
LDFLAGSMPIOMP=-lm -lmpi -lgomp

SRC= dgif_lib.c \
	egif_lib.c \
	gif_err.c \
	gif_font.c \
	gif_hash.c \
	gifalloc.c \
	utils.c \
	main.c \
	openbsd-reallocarray.c \
	quantize.c

OBJ= $(OBJ_DIR_CUDA)/dgif_lib.o \
	$(OBJ_DIR_CUDA)/egif_lib.o \
	$(OBJ_DIR_CUDA)/gif_err.o \
	$(OBJ_DIR_CUDA)/gif_font.o \
	$(OBJ_DIR_CUDA)/gif_hash.o \
	$(OBJ_DIR_CUDA)/gifalloc.o \
	$(OBJ_DIR_CUDA)/utils.o \
	$(OBJ_DIR_CUDA)/main.o \
	$(OBJ_DIR_CUDA)/openbsd-reallocarray.o \
	$(OBJ_DIR_CUDA)/quantize.o \
	$(OBJ_DIR_CUDA)/test_filter_cuda.o

OBJ_MPI_OMP= $(OBJ_DIR_MPI_OMP)/dgif_lib.o \
	$(OBJ_DIR_MPI_OMP)/egif_lib.o \
	$(OBJ_DIR_MPI_OMP)/gif_err.o \
	$(OBJ_DIR_MPI_OMP)/gif_font.o \
	$(OBJ_DIR_MPI_OMP)/gif_hash.o \
	$(OBJ_DIR_MPI_OMP)/gifalloc.o \
	$(OBJ_DIR_MPI_OMP)/utils.o \
	$(OBJ_DIR_MPI_OMP)/main.o \
	$(OBJ_DIR_MPI_OMP)/openbsd-reallocarray.o \
	$(OBJ_DIR_MPI_OMP)/quantize.o

all: $(OBJ_DIR_CUDA) sobelf_cuda $(OBJ_DIR_MPI_OMP) sobelf_mpi_omp

$(OBJ_DIR_CUDA):
	mkdir $(OBJ_DIR_CUDA)
$(OBJ_DIR_MPI_OMP):
	mkdir $(OBJ_DIR_MPI_OMP)

$(OBJ_DIR_CUDA)/test_filter_cuda.o: $(SRC_DIR)/test_filter_cuda.cu $(OBJ_DIR_CUDA)
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

$(OBJ_DIR_CUDA)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -fopenmp -o $@ $^
$(OBJ_DIR_MPI_OMP)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -fopenmp -o $@ $^

sobelf_cuda:$(OBJ)
	$(CC) $(NVCFLAGS) -o $@ $^ $(LDFLAGS)
sobelf_mpi_omp:$(OBJ_MPI_OMP)
	$(CC) $(NVCFLAGS) -o $@ $^ $(LDFLAGSMPIOMP)

clean:
	rm -f sobelf_cuda $(OBJ)
	rm -f sobelf_mpi_omp $(OBJ_MPI_OMP)