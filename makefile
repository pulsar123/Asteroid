#OPT=-G -g -DGPU -DSIMPLEX -arch=sm_60 --ptxas-options=-v
OPT=-O2 -DGPU -DSIMPLEX -arch=sm_60
#OPT=-g -O0
INC=-I/usr/include/cuda -I.


objects = asteroid.o read_data.o chi2.o misc.o cuda.o gpu_prepare.o

all: $(objects)
	nvcc $(OPT)  $(objects) -o ../asteroid

%.o: %.c
	nvcc $(OPT) -x cu  $(INC) -dc $< -o $@

clean:
	rm -f *.o ../asteroid
