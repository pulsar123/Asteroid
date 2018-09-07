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

// Precision for CUDA chi^2 calculations (float or double):
#define CHI_FLOAT float

// If defined, writes a file with delta V (geometry corrections for the observational data)
//#define DUMP_DV

// If defined, writes red_blue_corr.dat file for Drahus et al data corrected for geometry and time travel
//#define DUMP_RED_BLUE

// Total number of parameters (inpendent and dependent):
#ifdef BC
const int N_PARAMS = 10;
#else
const int N_PARAMS = 8;
#endif

// Number of independent parameters:
const int N_INDEPEND = 5;

// Maximum number of filters:
const int N_FILTERS = 1;

// When b and c parameters are used, maximum ln deviation from corresponding b_tumb, c_tumb during optimization:
const float BC_DEV_MAX = 100;  //2.3
// The same, but only during the initial value generation (when RANDOM_BC option is used):
const float BC_DEV1 = 1.1; //1.3


// GPU optimization parameters:
const int BSIZE = 256;   // Threads in a block (64 ... 1024, step of 64); 384; 256
#ifdef DEBUG
const int N_BLOCKS = 56*1;
#else
const int N_BLOCKS = 56*10; // Should be proportional to the number of SMs (56 for P100); code runtime and memory consumed on GPU is proportional to this number; x1000 for 1 day
#endif
//const int N_SERIAL = 1; // number of serial iglob loops inside the kernel (>=1)
//const int N_WARPS = BSIZE / 32;

// ODE time step (days):
const double TIME_STEP = 1e-2;  // 1e-2

// Simplex parameters:
#if defined(TIMING) || defined(DEBUG)
const unsigned int N_STEPS = 100; 
#else
const unsigned int N_STEPS = 7500; // Number of simplex steps per CUDA block (per simplex run) 27,000 per hour (N=7; BS=256; NB=56*4)
#endif
#ifdef DEBUG
const unsigned int DT_DUMP = 30;
#else
const unsigned int DT_DUMP = 300; // Time in seconds between results dump (to stdout)
#endif
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
//const int MAX_FILTERS = 100;
// Maximum number of data points:
const int MAX_DATA = 400;

// Number of time points for plotting
const int NPLOT = 6000; // 6000 !!!
// Times BSIZE will give the total number of points for lines:
const int C_POINTS = 10;
// Maximum relative deviation for each parameter when computing lines:
const double DELTA_MAX = 0.001;
// Scales for the V and t axes when computing the 2D least squares distances between the data and model:
const double V_SCALE = 1.0;  // in magnitudes
const double T_SCALE = 0.06;  // in days

// Only matter for REOPT option:
// Minimum and maximum initial simplex step:
const CHI_FLOAT DX_MIN = -9.2; // log(0.0001)
const CHI_FLOAT DX_MAX = -2.3; // log(0.1) -3.51
// Initial point is randomly shifted along each dimension by maximum 1/2 of the following amount (dimensionless):
const CHI_FLOAT DX_RAND = 0.1; 

// Maximum number of clusters in minima() periodogram search
const int NCL_MAX = 5;


// Speed of light (au/day):
const double light_speed = 173.144632674;

// Empirical coefficients for P_phi constraining, for LAM=0 and 1 cases:
const double S_LAM0 = 1.1733;
const double S_LAM1 = 1.2067;

// Parameters structure:
struct parameters_struct {
    // Independent parameters:
    double theta_M; // (angle between barycentric Z axis and angular momentum vector M); range 0...pi
    double phi_M;   // (polar angle for the angular momentum M in the barycentric FoR); range 0 ... 2*pi
    double phi_0;   // (initial Euler angle for precession); 0...2*pi
    double L;       // Angular momentum L value, radians/day; if P is perdiod in hours, L=48*pi/P
    double c_tumb;  // physical (tumbling) value of the axis c size; always smallest
    // Dependent parameters:
    double b_tumb;  // physical (tumbling) value of the axis b size; c_tumb < b_tumb < 1
    double Es;      // dimensionless total energy (asteroid's excitation degree), constrained by b_tumb, c_tumb
    double psi_0;   // initial Euler angle of rotation of the body, constrained by b_tumb, c_tumb, Es
    double c;       // Ellipsoid axis c/a size, for brightness purposes
    double b;       // Ellipsoid axis b/a size, for brightness purposes
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
int read_data(char *, int *, int *, int);
int chi2 (int, int, struct parameters_struct, double *);
int quadratic_interpolation(double, double *,double *,double *, double *,double *,double *);
int timeval_subtract (double *, struct timeval *, struct timeval *);
__device__ __host__ void iglob_to_params(int *, struct parameters_struct *);
__device__ __host__ void iloc_to_params(long int *, struct parameters_struct *);
int cmpdouble (const void * a, const void * b);
int minima(struct obs_data * dPlot, double * Vm, int Nplot);

#ifdef GPU

int gpu_prepare(int, int, int, int);

__global__ void setup_kernel ( curandState *, unsigned long, CHI_FLOAT *);
__global__ void chi2_gpu(struct obs_data *, int, int, curandState*, CHI_FLOAT*, struct parameters_struct*);
__global__ void chi2_plot(struct obs_data *, int, int,
                          struct parameters_struct *, struct obs_data *, int, struct parameters_struct, double *);
  #ifdef DEBUG2
  __global__ void debug_kernel(struct parameters_struct, struct obs_data *, int, int);
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

EXTERN char all_filters[N_FILTERS];


EXTERN struct obs_data *hData;
EXTERN struct obs_data *hPlot;

// CUDA version of the data:
EXTERN struct obs_data *dData;
EXTERN struct obs_data *dPlot;


// Arrays used for ephemerides interpolation:
EXTERN double E_x0[3],E_y0[3],E_z0[3], S_x0[3],S_y0[3],S_z0[3], MJD0[3];    
EXTERN double *MJD_obs;  // observational time (with light delay)
EXTERN double hMJD0;

EXTERN CHI_FLOAT * d_chi2_min;
EXTERN CHI_FLOAT * h_chi2_min;
EXTERN double * d_dlsq2;
EXTERN double * h_dlsq2;
EXTERN long int * d_iloc_min;
EXTERN long int * h_iloc_min;

EXTERN __device__ CHI_FLOAT d_chi2_plot;
EXTERN CHI_FLOAT h_chi2_plot;
    EXTERN __device__ CHI_FLOAT dLimits[2][N_INDEPEND];
    EXTERN __device__ double d_Vmod[NPLOT];
    EXTERN double h_Vmod[NPLOT];
EXTERN __device__ CHI_FLOAT d_chi2_lines[N_PARAMS][BSIZE*C_POINTS];
EXTERN CHI_FLOAT h_chi2_lines[N_PARAMS][BSIZE*C_POINTS];
    EXTERN CHI_FLOAT *d_f;
    EXTERN struct parameters_struct *d_params;
    EXTERN __device__ struct parameters_struct d_params0;
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

    EXTERN double cl_fr[NCL_MAX];
    EXTERN double cl_H[NCL_MAX];
EXTERN __device__ float dPphi;

#endif




