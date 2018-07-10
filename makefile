OPT=-g -DDEBUG -DGPU -DRELAXED -DBC -DP_PSI -DMINIMA_SPLINE -DMINIMA_PRINT -DPARABOLIC_MAX -arch=sm_60
INC=-I/usr/include/cuda -I.

BINARY=asteroid2

objects = asteroid.o read_data.o misc.o cuda.o gpu_prepare.o

all: $(objects)
	nvcc $(OPT)  $(objects) -o ../$(BINARY)

%.o: %.c
	nvcc $(OPT) -x cu  $(INC) -dc $< -o $@

clean:
	rm -f *.o ../$(BINARY)
