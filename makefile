OPT=-O2


objects = asteroid.o read_data.o chi2.o misc.o

all: $(objects)
	nvcc $(OPT) -arch=sm_20 $(objects) -o ../asteroid

%.o: %.c
	nvcc $(OPT) -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
	rm -f *.o ../asteroid
