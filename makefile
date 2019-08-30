# Macro parameters:

# ACC : enable high accuracy mode (mainly for final reoptimization): makes CHI_FLOAT=double, and reduces SIZE_MIN to 1e-10
# ANIMATE : produce animation of the projected atseroid rotation (a sequence of image files)
# BC : if defined, "physical b,c" and "photometric b,c" are independent parameters; if not, they are the same thing
# BW_BALL : simplest albedo (non-geometric) brightness model - black and white ball. Three new parameters: theta_R, phi_R, (theta_h, phi_h in paper) and kappa.
# DEBUG : used with interactive (debugging) runs, reduced kernels and print time intervals
# DUMP_DV : dumping 5.0*log10(1.0/E * 1.0/S) in read_data.c for all obs. data points
# DUMP_RED_BLUE : dumping the converted/corrected obs. data (MJD, V, w)
# INTERP : doing E,S vectors interpolation on GPU - slower, but can use many more data points (>490)
# LAST : (only for TORQUE) when -plot is used, printing the final values of the model parameters (L and E)
# MIN_DV : force certain minimum for dV (magnitudes) of the brightness curve
# MINIMA_PRINT : dumping periodogramm (fr, H) as min_profile.dat, in misc.c
# MINIMA_SPLINE : if defined, use spline-smoothed method to compute the periodogramm (only used with MINIMA_PRINT)
# MINIMA_TEST : (only works in -plot mode); test of how likely disk vs cigar models can produce minima as deep as observed (reshuffles theta_M, phi_M, phi_0 params)
# MY_L : input and output L values are not L, but 48*pi/L (purely for historical reasons)
# NO_SDATA : don't created shared memory sData array, use directly the device memory version
# NOPRINT : if defined, do not create files model.dat, data.dat, lines.dat
# NUDGE : nudging the model minima towards the observed minima (in 2D - t,V coordinates) during optimization (not working with SEGMENT)
# ONE_LE : only in SEGMENT mode: makes L,Es parameters multi-segment (fixed across all segments)
# PARABOLIC_MAX : use more accurate (parabola) method to find brightness minima when doing periodogramm (only with MINIMA_PRINT)
# P_BOTH : combined P_psi and P_phi constraints (input args: P_psi1 P_psi2 P_phi). P_Psi is a free parameter, P_phi is a constant. A rejection method is used during optimization.
# P_PHI : if defined, Pphi1 Pphi2 args need to be provided; L is no longer an input parameter, and is computed from P_phi using an approximate empirical relation
# P_PSI : if defined, Ppsi1 Ppsi2 args need to be provided; L is no longer an input parameter, and is computed precisely from P_psi
# PLOT_OMEGA : if defined and with -plot switch, will write file omega.dat with the time elvolution of Omega (angular velocity vector) in inertial spherical coordinates
# PROFILES : if defined, write cross-sections along all parameter dimensions to lines.dat
# RANDOM_BC : (not working) for BC mode. If defined, initial guess for brightness b,c parameters are random (not coinciding with the kinematic b,c parameters).
# RECT : rectangular prism simplified (phase=0) brightness model (here b, c parameters are half-lengths of the second and third shortest sides). Uses BC internally
# RMSD : confidence interval estimation using random point shifts around the input model
# ROTATE: only in BC mode; rotates the asteroid brightness frame relative to the inertia frame; three extra parameters: theta_R, phi_R, psi_R
# SEGMENT : multiple data segments (specified by T_START[] vector)
# SPHERICAL_K : (only makes sense when used with RMSD) : for confidence intervals calculations, use convert torque vector to spherical coordinates: r, theta, phi
# TIMING : time the main kernel (chi2_gpu)
# TORQUE : adding a simple constant torque model, with 3 extra parameters: Ti, Ts, Tl (same as Tb, Tc, Ta)
# TORQUE2 (implies TORQUE): torque parameters change half-way through the data (at mid-point in time). Adds 4 more parameters (theta_K2, phi_K2, phi_F2, K2)
# TREND : detrending the time evolution of the brightness, via the scaling parameter a (proxy for G-parameter from HG reflectivity law) - adds one free parameter A

ARCH=sm_60

OPT=--ptxas-options=-v -arch=$(ARCH) -DP_PSI -DTORQUE -DINTERP -DANIMATE
INC=-I/usr/include/cuda -I.
LIB=-lpng
DEBUG=-O2

BINARY=asteroid

objects = asteroid.o read_data.o misc.o cuda.o gpu_prepare.o

all: $(objects)
	nvcc $(OPT) $(DEBUG)  $(objects) -o ../$(BINARY)  ${LIB}

%.o: %.c makefile asteroid.h
	nvcc $(OPT) $(DEBUG) -x cu  $(INC) -dc $< -o $@

clean:
	rm -f *.o ../$(BINARY)

debug: DEBUG = -G -g -DDEBUG

debug: all

# grep "^#" *.c* *.h|cut -d# -f2|awk '{print $2}'|sort |uniq |grep -v "\.h"
