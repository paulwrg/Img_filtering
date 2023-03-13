SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
NVCC = nvcc
CFLAGS=-O3 -I$(HEADER_DIR) -std=gnu99 
NVCFLAGS=-O3 -I$(HEADER_DIR) 
LDFLAGS=-lm -lmpi -lgomp  -L/usr/local/cuda/lib64 -lcudart

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

OBJ= $(OBJ_DIR)/dgif_lib.o \
	$(OBJ_DIR)/egif_lib.o \
	$(OBJ_DIR)/gif_err.o \
	$(OBJ_DIR)/gif_font.o \
	$(OBJ_DIR)/gif_hash.o \
	$(OBJ_DIR)/gifalloc.o \
	$(OBJ_DIR)/utils.o \
	$(OBJ_DIR)/main.o \
	$(OBJ_DIR)/openbsd-reallocarray.o \
	$(OBJ_DIR)/quantize.o \
	$(OBJ_DIR)/test_filter_cuda.o

all: $(OBJ_DIR) sobelf

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/test_filter_cuda.o: $(SRC_DIR)/test_filter_cuda.cu $(OBJ_DIR)
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -fopenmp -o $@ $^

sobelf:$(OBJ)
	$(CC) $(NVCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f sobelf $(OBJ)