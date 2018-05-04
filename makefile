OPT=-O2 -DGPU -DRELAXED -DBC -arch=sm_60
INC=-I/usr/include/cuda -I.

BINARY=asteroid4

objects = asteroid.o read_data.o misc.o cuda.o gpu_prepare.o

all: $(objects)
	nvcc $(OPT)  $(objects) -o ../$(BINARY)

%.o: %.c
	nvcc $(OPT) -x cu  $(INC) -dc $< -o $@

clean:
	rm -f *.o ../$(BINARY)
