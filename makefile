OPT=-O2 -DGPU
INC=-I/usr/include/cuda -I.


objects = asteroid.o read_data.o chi2.o misc.o cuda.o gpu_prepare.o

all: $(objects)
	nvcc $(OPT) -arch=sm_20 $(objects) -o ../asteroid

%.o: %.c
	nvcc $(OPT) -x cu -arch=sm_20 $(INC) -dc $< -o $@

clean:
	rm -f *.o ../asteroid
