# Macro parameters:

# BC : if defined, "physical b,c" and "photometric b,c" are independent parameters; if not, they are the same thing
# DARK_SIDE : not functional. An attempt to introducde a toy "white/black ball" model
# DEBUG : used with interactive (debugging) runs, reduced kernels and print time intervals
# DEBUG2 : likley not functional; used for cuda code debugging
# DUMP_DV : dumping 5.0*log10(1.0/E * 1.0/S) in read_data.c for all obs. data points
# DUMP_RED_BLUE : dumping the converted/corrected obs. data (MJD, V, w)
# GPU : likely obligatory; use GPU for computations
# KAPPA : not functional. Parameter used in DARK_SIDE mode
# LSQ : likely not functional. Computing 2D least squares distances between the data points and the model, in chi2_plot
# MINIMA_PRINT : dumping periodogramm (fr, H) as min_profile.dat, in misc.c
# MINIMA_SPLINE : if defined, use spline-smoothed method to compute the periodogramm (only used with MINIMA_PRINT)
# NOPRINT : if defined, do not create files model.dat, data.dat, lines.dat
# PARABOLIC_MAX : use more accurate (parabola) method to find brightness minima when doing periodogramm (only with MINIMA_PRINT)
# P_PHI : if defined, Pphi1 Pphi2 args need to be provided; L is no longer an input parameter, and is computed from P_phi using an approximate empirical relation
# P_PSI : if defined, Ppsi1 Ppsi2 args need to be provided; L is no longer an input parameter, and is computed precisely from P_psi
# PROFILES : if defined, write cross-sections along all parameter dimensions to lines.dat
# RELAXED : during optimization, the free parameteres has the least possible degree of constraint.
# REOPT : if defined, use the model point provided via args, and searhc for minima around it
# TIMING : time the main kernel (chi2_gpu)


OPT=-O2 -DGPU -DRELAXED -DBC -DP_PSI -DMINIMA_SPLINE -DMINIMA_PRINT -DPARABOLIC_MAX -arch=sm_60
INC=-I/usr/include/cuda -I.

BINARY=asteroid2

objects = asteroid.o read_data.o misc.o cuda.o gpu_prepare.o

all: $(objects)
	nvcc $(OPT)  $(objects) -o ../$(BINARY)

%.o: %.c
	nvcc $(OPT) -x cu  $(INC) -dc $< -o $@

clean:
	rm -f *.o ../$(BINARY)


# grep "^#" *.c* *.h|cut -d# -f2|awk '{print $2}'|sort |uniq |grep -v "\.h"
