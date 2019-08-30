/*  Header file for the ABC (Asteroid Brightness in CUDA) project
 */

#ifndef ASTEROID_H
#define ASTEROID_H
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include "cuda_errors.h"
#ifdef ANIMATE
  #include <png.h>
#endif

#define PI 3.141592653589793238
#define RAD (180.0 / PI)

#define HUGE 1e30

// Constants:

// Precision for CUDA chi^2 calculations (float or double):
#ifdef ACC
  #define CHI_FLOAT double
#else
  #define CHI_FLOAT float
#endif
// Precision for observational data (structure obs_data):
#define OBS_TYPE double

#ifdef TORQUE2
 #define TORQUE
#endif

#ifdef RECT
 #define BC
#endif 

#ifdef SEGMENT
// Absolute times - starting points of the data segments:
#if N_SEG == 3
 const double T_START[N_SEG]={58051.044623, 58053.078872, 58055.234144};
 #else
 const double T_START[N_SEG]={58051.044623, 58053.078872, 58054.093273, 58055.234144};
 #endif
#else
const int N_SEG=1;
#endif

#ifndef sData
#ifdef NO_SDATA
#define sData dData
#endif
#endif

// If defined, writes a file with delta V (geometry corrections for the observational data)
//#define DUMP_DV

// If defined, writes red_blue_corr.dat file for Drahus et al data corrected for geometry and time travel
//#define DUMP_RED_BLUE

// Total number of parameters (inpendent and dependent):
#ifdef BC
const int DN_BC = 2;
#else
const int DN_BC = 0;
#endif

#ifdef ROTATE
const int DN_ROT = 3;
#else
const int DN_ROT = 0;
#endif

#ifdef TREND
const int DN_TREND = 1;
#else
const int DN_TREND = 0;
#endif

#ifdef TORQUE
  #ifdef TORQUE2
  const int DN_TORQUE = 7;
  #else
  const int DN_TORQUE = 3;
  #endif
#else
const int DN_TORQUE = 0;
#endif

#ifdef ONE_LE
const int DN_LE = 2;
#else
const int DN_LE = 0;
#endif

#ifdef BW_BALL
const int DN_BW = 3;
#else
const int DN_BW = 0;
#endif

#ifdef SEGMENT
// In multi-segmented mode, two obligatory parameters (c_tumb, b_tumb) and three optional parameters (c, b, A) are fixed across all segments:
const int N_PARAMS = (6+DN_TORQUE)*N_SEG + 2 + DN_BC + DN_ROT + DN_TREND - DN_LE*(N_SEG-1) + DN_BW;
// Number of parameters in a single (0-th) segment:
const int N_PARAMS0 = 8 + DN_BC + DN_ROT + DN_TREND + DN_TORQUE + DN_BW;
#else
const int N_PARAMS = 8 + DN_BC + DN_ROT + DN_TREND + DN_TORQUE + DN_BW;
#endif

//-------------------------- Property array -------------------------
// Names for the columns of the Property table:
const int P_type           = 0;  // The type of the parameter (starts with T_; e.g. T_phi_M; see below)
const int P_independent    = 1;  // if 1, "independent" parameter (limits are fixed, and described in hLimits array in asteroid.c); if 0, limits are not fixed, and depend on other parameters
const int P_frozen         = 2;  // if 1, the parameter is frozen (does not change during optimization), if 0, it is optimizable
const int P_iseg           = 3;  // in SEGMENT mode, containes the segment index iseg (in non-SEGMENT mode, always 0)
const int P_multi_segment  = 4;  // in SEGMENT mode, if 1 fixes the value of the parameter across all the segments; if 0, each segment gets its own optimizable parameter of this type
const int P_periodic       = 5;  // phi-like parameter (periodic, changes between 0 an 2*pi)

// Number of the columns of the Property table:
const int N_COLUMNS = 6;

// Values for the P_periodic column:
const int PERIODIC = 1;  // always periodic
const int PERIODIC_LAM = 0;  // only periodic when LAM (psi_0)
const int SOFT_BOTH = -1;  // both limits are soft
const int HARD_LEFT = -2;  // left limit is hard
const int HARD_RIGHT = -3;  // right limit is hard
const int HARD_BOTH = -4;  // both limits are hard

//char Type_names[20]; // Up to 50 free parameter types, up to 20 characters each

// Parameter type constants:
// Using non-standard macro parameter __COUNTER__ (increments by 1 every time it's called; works under gcc and icc)
const int T_theta_M = __COUNTER__;  
//#define NAME1 T_theta_M
//const int NAME1 = __COUNTER__; 
//strcpy(Type_names, "test");

const int T_phi_M =   __COUNTER__;
const int T_phi_0 =   __COUNTER__;
const int T_L =       __COUNTER__;
#ifdef TREND
const int T_A =       __COUNTER__;
#endif
#ifdef TORQUE
const int T_Ti =      __COUNTER__;
const int T_Ts =      __COUNTER__;
const int T_Tl =      __COUNTER__;
#endif
#ifdef TORQUE2
const int T_T2i =      __COUNTER__;
const int T_T2s =      __COUNTER__;
const int T_T2l =      __COUNTER__;
const int T_Tt  =      __COUNTER__;
#endif
const int T_c_tumb =  __COUNTER__;  //4
const int T_b_tumb =  __COUNTER__;  //5
const int T_Es =      __COUNTER__;
const int T_psi_0 =   __COUNTER__;
#ifdef BC
const int T_c =       __COUNTER__;  //8
const int T_b =       __COUNTER__;  //9
#endif
#if defined(ROTATE) || defined(BW_BALL)
const int T_theta_R = __COUNTER__;
const int T_phi_R =   __COUNTER__;
#endif
#ifdef ROTATE
const int T_psi_R =   __COUNTER__;
#endif
#ifdef BW_BALL
const int T_kappa =   __COUNTER__;
#endif
// Total number of parameter types (determines the length of the Limits and Types arrays):
const int N_TYPES =   __COUNTER__;

//-----------------------------------------------------------------------


// Maximum number of data points:
#ifdef INTERP
// INTERP function results in a ~15% slower code, but allows for a few time larger datasets:
const int MAX_DATA = 772; // 497, 662, 772, 993, 1914
#else
const int MAX_DATA = 390;
#endif
// Maximum number of filters:
const int N_FILTERS = 1;

// GPU optimization parameters:
const int BSIZE = 256;   // Threads in a block (64 ... 1024, step of 64); 256
#ifdef DEBUG
const int N_BLOCKS = 14;
#else
const int N_BLOCKS = 56; // Should be proportional to the number of SMs (56 for P100, 80 for V100)
#endif

// ODE time step (days):
const double TIME_STEP = 1e-2;  // 1e-2 for Oumuamua; 0.003 for TD60_All

// Simplex parameters:
const CHI_FLOAT DX_INI = 0.001;  // Maximum scale-free initial step
const CHI_FLOAT D2X_INI = -1.2; // Initial step size is log-random, in the interval exp(D2X_INI)*DX_INI .. DX_INI
// Only matter for reopt option:
// Initial point is randomly shifted along each dimension by maximum 1/2 of the following amount (dimensionless):
const CHI_FLOAT DX_RAND = 0.0001; 
#if defined(TIMING) || defined(DEBUG)
const unsigned int N_STEPS = 100; 
#else
const unsigned int N_STEPS = 5000; // Number of simplex steps per CUDA block (per simplex run) 27,000 per hour (N=7; BS=256; NB=56*4)
#endif

#ifdef NUDGE
const int M_MAX = 30;  // Maximum number of model local minima
const float M_MAX2 = 20;  // Soft limit on the number of local model minima (should be between NOBS_MAX and M_MAX); if M>N_MAX2, we start to punsih chi2
const int NOBS_MAX = 10;  // Maximum number of observed minima
const float DT_MAX = 0.12;  // Maximum 1D distance between observed and model minima in days; 0.12
const float DV_MAX = 24;  // Maximum 1D distance between observed and model minima in brightness magnitudes; 2.4
const float D2_MAX = sqrt(2)*DT_MAX;  // Maximum 2D distance between observed and model minima in equivalent days
const float DT_MAX2 = 1.5 * DT_MAX; // Additional multipler for DT_MAX defining the time window size (relative to observed minima) where model minima are memorized
const float P_MIN = 0.01;  // Reward strength for closeness of model minima to observed ones; 0...1; ->0 is the strongest reward
const float CHI2_0 = 10; // Below this value of chi2a, P_tot reward is fully applied
const float CHI2_1 = 30; // Above this value of chi2a, P_tot reward is not applied. The CHI2_0 ... CHI2_1 is the transition zone
const float L_RC = 0.05; // Lorentzian core radius for the nudge function, range 0...1
const float L_RC2 = L_RC * L_RC; // Derived parameter
const float L_A = 1.0/(1.0-L_RC2/(1.0+L_RC2));  // Lorentzian scale parameter
const float P_MIN2 = 1/P_MIN - 1;  // derived parameter
#endif

#ifdef ACC
const CHI_FLOAT SIZE_MIN = 1e-10; // Scale-free smallest simplex size (convergence criterion)
#else
const CHI_FLOAT SIZE_MIN = 1e-5; // Scale-free smallest simplex size (convergence criterion)
#endif
// Dimensionless simplex constants:
const CHI_FLOAT ALPHA_SIM = 1.0;
const CHI_FLOAT GAMMA_SIM = 2.0;
const CHI_FLOAT RHO_SIM   = 0.5;
const CHI_FLOAT SIGMA_SIM = 0.5;


const CHI_FLOAT SIZE2_MIN = SIZE_MIN * SIZE_MIN;

// When b and c parameters are used, maximum ln deviation from corresponding b_tumb, c_tumb during optimization:
const float BC_DEV_MAX = 100;  //2.3
// The same, but only during the initial value generation (when RANDOM_BC option is used):
const float BC_DEV1 = 1.1; //1.3


// Maximum number of chars in a file name:
const int MAX_FILE_NAME = 256;
// Maximum number of chars in one line of a data file:
const int MAX_LINE_LENGTH = 128;

// Pixel size of the square image (used in ANIMATE):
const int SIZE_PIX = 1080;

// Number of time points for plotting
#ifdef ANIMATE
const int dNPLOT = 1500;  // Number of snapshots produced per kernel; make it smaller if d_rgb array doesn't fit in GPU memory
const int NPLOT = 15440; // Total number of snapshots to produce ; 2700
#else
const int NPLOT = 20000; // 6000 !!!
#endif
// Times BSIZE will give the total number of points for lines:
const int C_POINTS = 10; // 10
// Maximum relative deviation for each parameter when computing lines:
const double DELTA_MAX = 0.01;
// Scales for the V and t axes when computing the 2D least squares distances between the data and model:
const double V_SCALE = 1.0;  // in magnitudes
const double T_SCALE = 0.06;  // in days

// Maximum number of clusters in minima() periodogram search
const int NCL_MAX = 5;

#ifdef MINIMA_TEST
const int N_PHI_0 = 256; // Number of phi_0 values (also threads in the kernel); ~256
const int N_THETA_M = 256; // Number of theta_M values (also x-dimension of the grid of blocks); should be an even number
const int N_PHI_M = 256; // Number of phi_M values (also y-dimension of the grid of blocks)
const int MAX_MINIMA = 50;  // Maximum number of brifhtness minima allowed during minima test
#endif

// Speed of light (au/day):
const double light_speed = 173.144632674;

// Empirical coefficients for P_phi constraining, for LAM=0 and 1 cases:
const double S_LAM0 = 1.1733;
const double S_LAM1 = 1.2067;

#ifdef MIN_DV
const float PV_MIN = 0.1; // chi2 reduction factor for large enough brightness fluctuations
const float DV_MIN1 = 2.25; // delta V at which chi2 starts decreasing (merit function goes from 1 to PV_MIN)
const float DV_MIN2 = 2.75; // delta V at which chi2 stops decreasing (merit function reached PV_MIN)
const float DV_MARGIN = 0.05; // Margins on both sides of the obs. data where min/max is not computed (days)
#endif

#ifdef ANIMATE
// All RGB values should be in [0,255] interval
// RGB values for deepest shadow:
const float SMIN_R = 6;
const float SMIN_G = 8;
const float SMIN_B = 10;
// RGB values for terminator:
const float SMAX_R = 30;
const float SMAX_G = 41;
const float SMAX_B = 53;
// RGB values for the brightest sunlit element:
const float IMAX_R = 255;
const float IMAX_G = 191;
const float IMAX_B = 144;
// Radius of the spot at the axis b end, scale-free units:
const float SPOT_RAD = 0.04;
#endif

// Structure to bring auxilary parameters to chi2one
struct chi2_struct {
    #ifdef NUDGE
    float t_obs[NOBS_MAX];
    float V_obs[NOBS_MAX];
    int N_obs;
    #endif   
    #ifdef SEGMENT
    int start_seg[N_SEG];
    #endif
    #ifdef INTERP
    double E_x0[3];
    double E_y0[3];
    double E_z0[3];
    double S_x0[3];
    double S_y0[3];
    double S_z0[3];
    double MJD0[3];
    #endif
};

// Structure used to pass parameters to x2params (from chi2gpu)
struct x2_struct {
    int reopt;
    #ifdef P_BOTH
    float Pphi;
    float Pphi2;
    #endif    
    // ???? Also for P_BOTH?
    #ifdef P_PSI
    float Ppsi1;
    float Ppsi2;
    #endif    
};


// Observational data arrays:
struct obs_data {
    float V;  // visual magnitude array, mag
    float w;  // 1-sgm error bar squared for V array, mag
    #ifndef INTERP    
    OBS_TYPE E_x;  // asteroid->Earth vector in barycentric FoR array, au
    OBS_TYPE E_y;  // asteroid->Earth vector in barycentric FoR array, au
    OBS_TYPE E_z;  // asteroid->Earth vector in barycentric FoR array, au
    OBS_TYPE S_x;  // asteroid->Sun vector in barycentric FoR array, au
    OBS_TYPE S_y;  // asteroid->Sun vector in barycentric FoR array, au
    OBS_TYPE S_z;  // asteroid->Sun vector in barycentric FoR array, au
    #endif
    OBS_TYPE MJD;  // asteroid time (without time delay)
    int Filter;  // Filter code array
};

#ifdef INTERP
struct obs_data_h {
    OBS_TYPE E_x;  // asteroid->Earth vector in barycentric FoR array, au
    OBS_TYPE E_y;  // asteroid->Earth vector in barycentric FoR array, au
    OBS_TYPE E_z;  // asteroid->Earth vector in barycentric FoR array, au
    OBS_TYPE S_x;  // asteroid->Sun vector in barycentric FoR array, au
    OBS_TYPE S_y;  // asteroid->Sun vector in barycentric FoR array, au
    OBS_TYPE S_z;  // asteroid->Sun vector in barycentric FoR array, au
};
#endif

// Function declarations
int read_data(char *, int *, int *, int);
int quadratic_interpolation(double, OBS_TYPE *,OBS_TYPE *,OBS_TYPE *, OBS_TYPE *,OBS_TYPE *,OBS_TYPE *);
int timeval_subtract (double *, struct timeval *, struct timeval *);
int cmpdouble (const void * a, const void * b);
int minima(struct obs_data * dPlot, double * Vm, int Nplot);
int prepare_chi2_params(int *);
int gpu_prepare(int, int, int, int);
int minima_test(int, int, int, double*, int[][N_SEG], CHI_FLOAT);

__global__ void setup_kernel ( curandState *, unsigned long, CHI_FLOAT *, int);
#ifndef ANIMATE
__global__ void chi2_gpu(struct obs_data *, int, int, int, int, curandState*, CHI_FLOAT*, double*, double*);
#ifdef RMSD
__global__ void chi2_gpu_rms(struct obs_data *, int, int, int, int, curandState*, CHI_FLOAT*, double*, double*, float, float*, float*);
#endif
#endif
__global__ void chi2_plot(struct obs_data *, int, int, struct obs_data *, int, double *,
#ifdef ANIMATE
                          unsigned char *,
#endif
                          float);
#ifdef MINIMA_TEST
__global__ void chi2_minima(struct obs_data *, int, int, struct obs_data *, int, CHI_FLOAT);
#endif
#ifdef DEBUG2
__global__ void debug_kernel(struct parameters_struct, struct obs_data *, int, int);
#endif
#ifdef ANIMATE
__device__ void compute_rgb(unsigned char *, double, double, 
                            double, double, double,
                            double, double, double,
                            double, double, double,
                            double, double, double,
                            double, double, double,
                            double, double, double,
                            int, int);
int write_PNGs(unsigned char *, int, int);
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
#ifdef INTERP
EXTERN __device__ double dE_x0[3],dE_y0[3],dE_z0[3], dS_x0[3],dS_y0[3],dS_z0[3], dMJD0[3];    
#endif

EXTERN CHI_FLOAT * d_chi2_min;
EXTERN CHI_FLOAT * h_chi2_min;
EXTERN double * d_dlsq2;
EXTERN double * h_dlsq2;
EXTERN long int * d_iloc_min;
EXTERN long int * h_iloc_min;

EXTERN __device__ CHI_FLOAT d_chi2_plot;
EXTERN CHI_FLOAT h_chi2_plot;
EXTERN __device__ CHI_FLOAT dLimits[2][N_TYPES];
EXTERN __device__ double d_Vmod[NPLOT];
#ifdef PLOT_OMEGA
EXTERN __device__ double d_Omega[6][NPLOT];
EXTERN double h_Omega[6][NPLOT];
#endif
EXTERN double h_Vmod[NPLOT];
#ifdef PROFILES
EXTERN __device__ CHI_FLOAT d_chi2_lines[N_PARAMS][BSIZE*C_POINTS];
EXTERN CHI_FLOAT h_chi2_lines[N_PARAMS][BSIZE*C_POINTS];
EXTERN __device__ CHI_FLOAT d_param_lines[N_PARAMS][BSIZE*C_POINTS];
EXTERN CHI_FLOAT h_param_lines[N_PARAMS][BSIZE*C_POINTS];
#endif
EXTERN CHI_FLOAT *d_f;

//EXTERN struct parameters_struct *d_params;
//EXTERN __device__ struct parameters_struct d_params0;
//EXTERN double __device__ d_params[N_BLOCKS][N_PARAMS];
EXTERN double* d_params;
//EXTERN double __device__ d_dV[N_BLOCKS][N_FILTERS];
EXTERN double* d_dV;
//EXTERN double h_params[N_BLOCKS][N_PARAMS];
EXTERN double* h_params;
//EXTERN double h_dV[N_BLOCKS][N_FILTERS];
EXTERN double* h_dV;
EXTERN __device__ double d_params0[N_PARAMS];
#ifdef RMSD
EXTERN float *dpar_min, *dpar_max;
EXTERN float *hpar_min, *hpar_max;
EXTERN __device__ int d_Ntot, d_Nbad;
EXTERN __device__ float d_f0, d_f1;
#endif

EXTERN __device__ unsigned long long int d_sum;
EXTERN __device__ unsigned long long int d_sum2;
EXTERN __device__ int d_min;
EXTERN __device__ int d_max;
EXTERN __device__ unsigned int d_block_counter;
EXTERN __device__ CHI_FLOAT d_delta_V[N_FILTERS];
EXTERN CHI_FLOAT h_delta_V[N_FILTERS];
EXTERN CHI_FLOAT *h_f;
EXTERN unsigned long long int h_sum;
EXTERN unsigned long long int h_sum2;
EXTERN int h_min;
EXTERN int h_max;
EXTERN unsigned int h_block_counter;

EXTERN double cl_fr[NCL_MAX];
EXTERN double cl_H[NCL_MAX];
EXTERN __device__ int dProperty[N_PARAMS][N_COLUMNS];
EXTERN __device__ int dTypes[N_TYPES][N_SEG];

#ifdef NUDGE
EXTERN __device__ struct chi2_struct d_chi2_params;
#endif

#ifdef SEGMENT
EXTERN int h_start_seg[N_SEG];
EXTERN int h_plot_start_seg[N_SEG];
EXTERN __device__ int d_start_seg[N_SEG];
EXTERN __device__ int d_plot_start_seg[N_SEG];
#endif
#ifdef LAST
EXTERN __device__ double d_L_last, d_E_last;
#endif
#ifdef MINIMA_TEST
EXTERN __device__ float d_Scores[N_THETA_M][N_PHI_M];
EXTERN float h_Scores[N_THETA_M][N_PHI_M];
EXTERN __device__ float d_Prob[N_THETA_M][N_PHI_M];
EXTERN float h_Prob[N_THETA_M][N_PHI_M];
EXTERN __device__ int d_N7all;
EXTERN int h_N7all;
#endif
EXTERN __device__ struct x2_struct d_x2_params;

#ifdef ANIMATE
// Format: [rgb][snapshot][x][y]
EXTERN __device__ unsigned char * d_rgb;
EXTERN unsigned char * h_rgb;
EXTERN __device__ int d_i1, d_i2;
EXTERN int h_i1, h_i2;
#endif

#endif




