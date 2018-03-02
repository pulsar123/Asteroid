/*  Header file for the ABC (Asteroid Brightness in CUDA) project
 */

#ifndef ASTEROID_H
#define ASTEROID_H
#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include "cuda_errors.h"

#define PI 3.141592653589793238
#define RAD (180.0 / PI)

// Constants:
// Number of free parameters for chi^2 (excludes filters):
// &&&
#ifdef TUMBLE
const int N_PARAMS = 10;
#else
const int N_PARAMS = 7;
#endif

// Precision for CUDA chi^2 calculations (float or double):
#define CHI_FLOAT float

// Enforcing the c<b condition if defined:
//#define FORCE_BC
// Log scale for b and c if defined:
#define LOG_BC

// Maximum number of filters:
const int N_FILTERS = 11;

// Parameter #1: b/a
const double B1 = 0.1;
const double B2 = 0.1;
const int N_B = 1;

// Parameter #2: period P, days
const double P1 = 7.5/24;
const double P2 = 7.5/24;
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
const double PHI_A1 = 0.0;
const double PHI_A2 = 2.0*PI;
const int N_PHI_A = 360*4;

// GPU optimization parameters:
const int BSIZE = 256;   // Threads in a block (64 ... 1024, step of 64); 384
const int N_BLOCKS = 56*2000; // Should be proportional to the number of SMs (56 for P100); code runtime and memory consumed on GPU is proportional to this number; x1000 for 1 day
//const int N_SERIAL = 1; // number of serial iglob loops inside the kernel (>=1)
//const int N_WARPS = BSIZE / 32;

// Simplex parameters:
#ifdef TIMING
const unsigned int N_STEPS = 1000; 
#else
const unsigned int N_STEPS = 100000; // Number of simplex steps per CUDA block (per simplex run) 27,000 per hour (N=7; BS=256; NB=56*4)
#endif
const unsigned int DT_DUMP = 180; // Time in seconds between results dump (to stdout)
const int N_WRITE = 1; // Every N_WRITE dumps make a dump to results.dat file
const CHI_FLOAT DX_INI = 0.01;  // Scale-free initial step
const CHI_FLOAT SIZE_MIN = 1e-5; // Scale-free smallest simplex size (convergence criterion)
// Dimensionless simplex constants:
const CHI_FLOAT ALPHA_SIM = 1.0;
const CHI_FLOAT GAMMA_SIM = 2.0;
const CHI_FLOAT RHO_SIM   = 0.5;
const CHI_FLOAT SIGMA_SIM = 0.5;

const CHI_FLOAT SIZE2_MIN = SIZE_MIN * SIZE_MIN;

// Maximum number of chars in a file name:
const int MAX_FILE_NAME = 256;
// Maximum number of chars in one line of a data file:
const int MAX_LINE_LENGTH = 128;
// Maximum number of filters:
const int MAX_FILTERS = 100;
// Maximum number nof data points:
const int MAX_DATA = 400;

// Speed of light (au/day):
const double light_speed = 173.144632674;

// Parameters structure:
struct parameters_struct {
    double b;          // b/a 
    double P;          // internal period P (hours)
    double theta;      // angle between internal rotation axis (no TUMBLE) or precession axis (TUMBLE) and axis z
    double cos_phi;    // phi is the polar angle for internal rotation axis (no TUMBLE) or precession axis (TUMBLE)
    double phi_a0;     // initial polar angle for a orientation
    double c;          // c/a
    double cos_phi_b;  // initial polar angle for b orientation
#ifdef TUMBLE
    double P_pr;       // precession period (hours)
    double theta_pr;   // angle between internal rotation axis and precession axis
    double phi_n0;     // initial polar angle for the rotation axis
#endif    
};

// Observational data arrays:
struct obs_data {
float V;  // visual magnitude array, mag
float w;  // 1-sgm error bar squared for V array, mag
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
int chi2 (int, int, struct parameters_struct, double *);
int quadratic_interpolation(double, double *,double *,double *, double *,double *,double *);
int timeval_subtract (double *, struct timeval *, struct timeval *);
__device__ __host__ void iglob_to_params(int *, struct parameters_struct *);
__device__ __host__ void iloc_to_params(long int *, struct parameters_struct *);

#ifdef GPU

int gpu_prepare(int, int, int);

#ifdef SIMPLEX
__global__ void setup_kernel ( curandState *, unsigned long, CHI_FLOAT *);
__global__ void chi2_gpu(struct obs_data *, int, int, curandState*, CHI_FLOAT*, struct parameters_struct*);
#else
__global__ void chi2_gpu(struct obs_data *, int, int, long int, int, int, float*, long int*);
#endif
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

EXTERN CHI_FLOAT * d_chi2_min;
EXTERN CHI_FLOAT * h_chi2_min;
EXTERN long int * d_iloc_min;
EXTERN long int * h_iloc_min;

#ifdef SIMPLEX
    EXTERN __device__ CHI_FLOAT dLimits[2][N_PARAMS];
    EXTERN CHI_FLOAT *d_f;
    EXTERN struct parameters_struct *d_params;
    EXTERN __device__ unsigned long long int d_sum;
    EXTERN __device__ unsigned long long int d_sum2;
    EXTERN __device__ int d_min;
    EXTERN __device__ int d_max;
    EXTERN __device__ unsigned int d_block_counter;
    EXTERN CHI_FLOAT *h_f;
    EXTERN struct parameters_struct *h_params;
    EXTERN unsigned long long int h_sum;
    EXTERN unsigned long long int h_sum2;
    EXTERN int h_min;
    EXTERN int h_max;
    EXTERN unsigned int h_block_counter;
#endif                


#endif




