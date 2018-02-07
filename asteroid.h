/*  Header file for the ABC (Asteroid Brightness in CUDA) project
 */

#ifndef ASTEROID_H
#define ASTEROID_H
#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include "cuda_errors.h"

#define PI 3.141592653589793238L
#define RAD (180.0L / PI)


// Constants:
// Number of free parameters for chi^2 (excludes filters):
const int N_PARAMS = 7;

// Parameter #1: b/a
const double B1 = 0.2;
const double B2 = 0.2;
const int N_B = 1;

// Parameter #2: period P, days
const double P1 = 7.46/24;
const double P2 = 7.46/24;
const int N_P = 1;

// Fixed spin vector n params (barycentric FoR)
// The parameters are such that a uniform distribution in both results in a uniform spin vector distribution on a unit sphere
// Parameter #3: theta is the angle between n and e_z; range [0...180]:
const double THETA1 = 0.001/RAD;
const double THETA2 = 180.0/RAD;
const int N_THETA = 100;

// Parameter #4:  Cosine of the polar angle phi, range [-1...1[
const double COS_PHI1 = -1.0;
const double COS_PHI2 = 0.999;
const int N_COS_PHI = 100;

// Parameter #5:  rotation (phase angle), phi_a; range [0,360[
const double PHI_A1 = 1.0;
const double PHI_A2 = 1.0;
const int N_PHI_A = 360*4;

// GPU optimization parameters:
const int BSIZE = 256;   // Threads in a block (64 ... 1024, step of 64)
const int N_BLOCKS = 56; // Should be proprtional to the number of SMs (56 for P100)
//const int N_SERIAL = 1; // number of serial iglob loops inside the kernel (>=1)
const int N_WARPS = BSIZE / 32;



// Maximum number of chars in a file name:
const int MAX_FILE_NAME = 256;
// Maximum number of chars in one line of a data file:
const int MAX_LINE_LENGTH = 128;
// Maximum number of filters:
const int MAX_FILTERS = 100;

// Speed of light (au/day):
const double light_speed = 173.144632674;
const double Pi = acos(-1.0);
const double Rad = 180.0 / Pi;

// Parameters structure:
struct parameters_struct {
    double b; 
    double c;
    double P;
    double theta;
    double cos_phi;
    double cos_phi_b;
    double phi_a0;
};

// Observational data arrays:
struct obs_data {
double V;  // visual magnitude array, mag
double w;  // 1-sgm error bar squared for V array, mag
double E_x;  // asteroid->Earth vector in barycentric FoR array, au
double E_y;  // asteroid->Earth vector in barycentric FoR array, au
double E_z;  // asteroid->Earth vector in barycentric FoR array, au
double S_x;  // asteroid->Sun vector in barycentric FoR array, au
double S_y;  // asteroid->Sun vector in barycentric FoR array, au
double S_z;  // asteroid->Sun vector in barycentric FoR array, au
double MJD;  // asteroid time (without time delay)
int Filter;  // Filter code array
};


// Function declarations
int read_data(char *, int *, int *);
int chi2 (int, int, double *);
int quadratic_interpolation(double, double *,double *,double *, double *,double *,double *);
int timeval_subtract (double *, struct timeval *, struct timeval *);
__device__ __host__ void iglob_to_params(int *, struct parameters_struct *);
__device__ __host__ void iloc_to_params(long int *, struct parameters_struct *);

#ifdef GPU

int gpu_prepare(int, int);

__global__ void chi2_gpu(struct obs_data *, int, int, long int, int, int, double *, long int *);
#endif



// Global variables
#ifdef MAIN
// If called from main(), global variables are defined:
#define EXTERN
#else
// Otherwise, global variables are declared with "extern"
#define EXTERN extern
#endif

EXTERN char all_filters[MAX_FILTERS];


EXTERN struct obs_data *hData;

// CUDA version of the data:
EXTERN struct obs_data *dData;


// Arrays used for ephemerides interpolation:
EXTERN double E_x0[3],E_y0[3],E_z0[3], S_x0[3],S_y0[3],S_z0[3], MJD0[3];    
EXTERN double *MJD_obs;  // observational time (with light delay)
EXTERN double hMJD0;

EXTERN double * d_chi2_min;
EXTERN double * h_chi2_min;
EXTERN long int * d_iloc_min;
EXTERN long int * h_iloc_min;


#endif




