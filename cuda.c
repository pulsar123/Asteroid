/* CUDA stuff.
 * Computing chi^2 on GPU for a given combination of free parameters
 * 
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include "asteroid.h"


__device__ void ODE_func (double y[], double f[], double mu[])
/* Three ODEs for the tumbling evolution of the three Euler angles, phi, theta, and psi.
   Derived in a manner similar to Kaasalainen 2001, but for the setup of Samarasinha and A'Hearn 1991
   (a > b > c; Il > Ii > Is; either a or c can be the axis of rotation). The setup of Kaasalainen 2001 results
   in large derivatives, and low accuracy and instability of ODEs for small I1.
   
   Here Ip=(1/Ii+1/Is)/2; Im=(1/Ii-1/Is)/2.
   */

{    
// mu[0,1,2] -> L, Ip, Im
//    double phi = y[0];
//    double theta = y[1];
//    double psi = y[2];
    
    // dphi/dt:
    f[0] = mu[0]*(mu[1]-mu[2]*cos(2.0*y[2]));
    
    // dtheta/dt:
    f[1] = mu[0]*mu[2]*sin(y[1])*sin(2.0*y[2]);
    
    // dpsi/dt (assuming Il=1):
    f[2] = cos(y[1])*(mu[0]-f[0]);
}



__device__ CHI_FLOAT chi2one(struct parameters_struct params, struct obs_data *sData, int N_data, int N_filters, CHI_FLOAT *delta_V, int Nplot)
// Computung chi^2 for a single model parameters combination, on GPU, by a single thread
{
    int i, m;
    double cos_alpha_p, sin_alpha_p, scalar_Sun, scalar_Earth, scalar;
    double cos_lambda_p, sin_lambda_p, Vmod, alpha_p, lambda_p;
    double Ep_x, Ep_y, Ep_z, Sp_x, Sp_y, Sp_z;
    CHI_FLOAT chi2a;
    double sum_y2[N_FILTERS];
    double sum_y[N_FILTERS];
    double sum_w[N_FILTERS];

//!!! Seems to speed up a bit ODE code: - but probably breaks  __syncthreads at the end of chi2_gpu?
//    __syncthreads();
    
    for (m=0; m<N_filters; m++)
    {
        sum_y2[m] = 0.0;
        sum_y[m] = 0.0;
        sum_w[m] = 0.0;
    }
    
    /*  Tumbling model description:
     
     Triaxial ellipsoid with physical axes a, b, c. a and c are extremal ,
     b is always intermediate. (c < b < a=1) Photometric a,b,c can be totally different (not implemented yet).
     
     The corresponding moments of inertia are Il (a), Ii (b), Is(c); Il < Ii < Is.
     
     Input parameters with fixed constraints (5): 
      - <M>: angular momentum vector described by params.theta_M and params.phi_M
      - phi_0: initial Euler angle for precession, 0...2*pi
      - L: angular momentum L value, radians/day; if P is perdiod in hours, L=48*pi/P
      - c_tumb: log10 of the physical (tumbling) value of the smallest axis c size; c < b < a=1
      
     Derived values:
      - Ii_inv = 1/Ii; inverse principle moment of inertia Ii
      - Is_inv = 1/Is; inverse principle moment of inertia Is
      
     Parameters which are constrained by other parameters (3):
      - b_tumb: log10 of the physical (tumbling) value of the intermediate axis b size; constrained by c: log10(b)=log10(c)...0
      - Es: dimensionless total energy, constrained by Ii: SAM: Es<1/Ii; LAM: Es>1/Ii
      - psi_0: initial Euler angle of rotation of the body, constrained (only for SAM) by Ii, Is, Einv=1/Es: psi_max=atan(sqrt(Ii*(Is-Einv)/Is/(Einv-Ii))); psi_0=[-psi_max,psi_max]
      
     Derived values:
      - theta_0: initial Euler nutation angle; 0 ... pi range, derived from Ii, Is, Es, psi_0
     
     Time integration to compute the three Euler angles (phi, psi, theta) values for all observed data points
      - Initial conditions: phi_0, psi_0, theta_0
      - Parameters needed: L, Ip=0.5*(Ii_inv+Is_inv); Im=0.5*(Ii_inv-Is_inv);  
      - time step - macro parameter TIME_STEP (days)
     
     */
        
    // We work in the inertial observed (Solar barycentric) frame of reference X, Y, Z.
    // By applying sequentially the three Euler angles of the asteroid (which we compute by solving the three ODEs numerically),
    // we derive the asteroid's internal axes (coinciding with b, c, a for x, y, z)
    // orienation in the barycentric frame of reference. This allows us to compute the orientation of the asteroid->sun and asteroid->earth
    // vectors in the asteroid's frame of reference, which is then used to compute it's apparent brightness for the observer.

    // Calculations which are time independent:            
    
    // In tumbling mode, the vector M is the angular momentum vector (fixed in the inertial - barycentric - frame of reference)
    // It is defined by the two angles (input parameters) - params.theta_M and params.phi_M
    // It's a unit vector
    double M_x = sin(params.theta_M)*cos(params.phi_M);
    double M_y = sin(params.theta_M)*sin(params.phi_M);
    double M_z = cos(params.theta_M);
        
    // In the new inertial frame of reference with the <M> vector being the z-axis, we arbitrarily choose the x-axis, <XM>, to be [y x M].
    // It is fixed in the inertial frame of reference, so can be computed here:
    // Made a unit vector
    double XM = sqrt(M_z*M_z+M_x*M_x);
    double XM_x = M_z / XM;
//    double XM_y = 0.0;
    double XM_z = -M_x / XM;
    // The third axis, YM, is derived as [M x XM]; a unit vector by design:
    double YM_x = M_y*XM_z;
    double YM_y = M_z*XM_x - M_x*XM_z;
    double YM_z = -M_y*XM_x;

    // We set Il (moment of inertia corresponding to the largest axis, a) to 1.
    // Shortest axis (c), largest moment of inertia:
    double Is = (1.0+params.b_tumb*params.b_tumb)/(params.b_tumb*params.b_tumb+params.c_tumb*params.c_tumb);
    // Intermediate axis (b), intermediate moment of inertia:
    double Ii = (1.0+params.c_tumb*params.c_tumb)/(params.b_tumb*params.b_tumb+params.c_tumb*params.c_tumb);
    double Is_inv = 1.0 / Is;
    double Ii_inv = 1.0 / Ii;
    
    // Now we have a=1>b>c, and Il=1<Ii<Is
    // Axis of rotation can be either "a" (LAM) or "c" (SAM)
    
//    double Einv = 1.0/params.Es;
    
    double mu[3];
    double Ip = 0.5*(Ii_inv + Is_inv);
    double Im = 0.5*(Ii_inv - Is_inv);
    mu[0] = params.L;
    mu[1] = Ip;
    mu[2] = Im;    
    
    // Initial Euler angles values:
    double phi = params.phi_0;
    // Initial value of the Euler angle theta is determined by other parameters:
    double theta = asin(sqrt((params.Es-1.0)/(sin(params.psi_0)*sin(params.psi_0)*(Ii_inv-Is_inv)+Is_inv-1.0)));    
    double psi = params.psi_0;
            
    // The loop over all data points    
    for (i=0; i<N_data; i++)
    {                                
        
        // Derive the three Euler angles theta, phi, psi here, by solving three ODEs numerically
        if (i > 0)
        {
            int N_steps;
            double h;
            
            // How many integration steps to the current (i-th) observed value, from the previous (i-1) one:
            // Forcing the maximum possible time step of TIME_STEP days (macro parameter), to ensure accuracy
            N_steps = (sData[i].MJD - sData[i-1].MJD) / TIME_STEP + 1;
            // Current equidistant time steps (h<=TIME_STEP):
            h = (sData[i].MJD - sData[i-1].MJD) / N_steps;
            
            double y[3];
           
            // Initial angles values = the old values, from the previous i cycle:
            y[0] = phi;
            y[1] = theta;
            y[2] = psi;
            
            // RK4 method for solving ODEs with a fixed time step h
            for (int l=0; l<N_steps; l++)
            {
                double K1[3], K2[3], K3[3], K4[3], f[3];
                
                ODE_func (y, K1, mu);
        
                int j;
                for (j=0; j<3; j++)
                    f[j] = y[j] + 0.5*h*K1[j];
                ODE_func (f, K2, mu);
        
                for (j=0; j<3; j++)
                    f[j] = y[j] + 0.5*h*K2[j];
                ODE_func (f, K3, mu);
        
                for (j=0; j<3; j++)
                    f[j] = y[j] + h*K3[j];
                ODE_func (f, K4, mu);
        
                for (j=0; j<3; j++)
                    y[j] = y[j] + 1/6.0 * h *(K1[j] + 2*K2[j] + 2*K3[j] + K4[j]);
            }
        

            // New (current) values of the Euler angles derived from solving the ODEs:
            phi = y[0];
            theta = y[1];
            psi = y[2];                    
        }                

        // At this point we know the three Euler angles for the current moment of time (data point) - phi, theta, psi.
        
        double cos_phi = cos(phi);
        double sin_phi = sin(phi);

        // Components of the node vector N=[M x a], derived by rotating vector XM towards vector YM by Euler angle phi
        // It is unit by design
        // Using XM_y = 0
        double N_x = XM_x*cos_phi + YM_x*sin_phi;
        double N_y =                YM_y*sin_phi;
        double N_z = XM_z*cos_phi + YM_z*sin_phi;
        
        // Vector p=[N x M]; a unit one
        double p_x = N_y*M_z - N_z*M_y;
        double p_y = N_z*M_x - N_x*M_z;
        double p_z = N_x*M_y - N_y*M_x;
        
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        
        // Vector of rotation <a> (the longest axes of the ellipsoid; x3; z; l) is derived by rotating <M> by Euler angle theta towards <p>,
        // with the node vector <N> being the rotation vector (Rodrigues formula); a unit vector
        double a_x = M_x*cos_theta + p_x*sin_theta;
        double a_y = M_y*cos_theta + p_y*sin_theta;
        double a_z = M_z*cos_theta + p_z*sin_theta;
        
        // Vector w=[a x N]; a unit one
        double w_x = a_y*N_z - a_z*N_y;
        double w_y = a_z*N_x - a_x*N_z;
        double w_z = a_x*N_y - a_y*N_x;

        double sin_psi = sin(psi);
        double cos_psi = cos(psi);
        
        // Second axis of the ellipsoid, b (x1; x; i); a unit vector; derived by rotating <N> by Euler angle psi towards <w>,
        // with vector <a> being the rotation axis
        double b_x = N_x*cos_psi + w_x*sin_psi;
        double b_y = N_y*cos_psi + w_y*sin_psi;
        double b_z = N_z*cos_psi + w_z*sin_psi;
        
        // Third ellipsoid axis c (x2; y; s) - the shortest one; c=[a x b]; unit vector by design
        double c_x = a_y*b_z - a_z*b_y;
        double c_y = a_z*b_x - a_x*b_z;
        double c_z = a_x*b_y - a_y*b_x;

        
        // Earth vector in the new (a,b,c) basis; according to my tests in brightness.c, the correct brightness curve is observed when
        // the following sequence is used: a, c, b, for Ep_x,y,z.
        Ep_x = a_x*sData[i].E_x + a_y*sData[i].E_y + a_z*sData[i].E_z;
        Ep_y = c_x*sData[i].E_x + c_y*sData[i].E_y + c_z*sData[i].E_z;
        Ep_z = b_x*sData[i].E_x + b_y*sData[i].E_y + b_z*sData[i].E_z;
        
        // Sun vector in the new (a,b,c) basis:-- should be (b,c,a)???
        Sp_x = a_x*sData[i].S_x + a_y*sData[i].S_y + a_z*sData[i].S_z;
        Sp_y = c_x*sData[i].S_x + c_y*sData[i].S_y + c_z*sData[i].S_z;
        Sp_z = b_x*sData[i].S_x + b_y*sData[i].S_y + b_z*sData[i].S_z;
        
        // Now that we converted the Earth and Sun vectors to the internal asteroidal basis (a,b,c),
        // we can apply the formalism of Muinonen & Lumme, 2015 to calculate the brightness of the asteroid.
        
        // The two scalars from eq.(12) of Muinonen & Lumme, 2015; assuming a=1
        scalar_Sun   = sqrt(Sp_x*Sp_x + Sp_y*Sp_y/(params.b_tumb*params.b_tumb) + Sp_z*Sp_z/(params.c_tumb*params.c_tumb));
        scalar_Earth = sqrt(Ep_x*Ep_x + Ep_y*Ep_y/(params.b_tumb*params.b_tumb) + Ep_z*Ep_z/(params.c_tumb*params.c_tumb));
        
        // From eq.(13):
        cos_alpha_p = (Sp_x*Ep_x + Sp_y*Ep_y/(params.b_tumb*params.b_tumb) + Sp_z*Ep_z/(params.c_tumb*params.c_tumb)) / (scalar_Sun * scalar_Earth);
        sin_alpha_p = sqrt(1.0 - cos_alpha_p*cos_alpha_p);
        alpha_p = atan2(sin_alpha_p, cos_alpha_p);
        
        // From eq.(14):
        scalar = sqrt(scalar_Sun*scalar_Sun + scalar_Earth*scalar_Earth + 2*scalar_Sun*scalar_Earth*cos_alpha_p);
        cos_lambda_p = (scalar_Sun + scalar_Earth*cos_alpha_p) / scalar;
        sin_lambda_p = scalar_Earth*sin_alpha_p / scalar;
        lambda_p = atan2(sin_lambda_p, cos_lambda_p);
        
        // Asteroid's model visual brightness, from eq.(10):
        //!!! Not using P(alpha) - single scattering phase function!
        Vmod = -2.5*log10(params.b_tumb*params.c_tumb * scalar_Sun*scalar_Earth/scalar * (cos(lambda_p-alpha_p) + cos_lambda_p +
        sin_lambda_p*sin(lambda_p-alpha_p) * log(1.0 / tan(0.5*lambda_p) / tan(0.5*(alpha_p-lambda_p)))));
                        
        if (Nplot > 0)
        {
            d_Vmod[i] = Vmod + delta_V[0]; //???
        }
        else
        {
            // Filter:
            int m = sData[i].Filter;
            // Difference between the observational and model magnitudes:
            double y = sData[i].V - Vmod;                    
            sum_y2[m] = sum_y2[m] + y*y*sData[i].w;
            sum_y[m] = sum_y[m] + y*sData[i].w;
            sum_w[m] = sum_w[m] + sData[i].w;
        }
        
    } // data points loop
    
    
    if (Nplot > 0)
        return 0.0;
    
    CHI_FLOAT chi2m;
    chi2a=0.0;    
    for (m=0; m<N_filters; m++)
    {
        // Chi^2 for the m-th filter:
        chi2m = sum_y2[m] - sum_y[m]*sum_y[m]/sum_w[m];
        chi2a = chi2a + chi2m;
        // Average difference Vdata-Vmod for each filter (used for plotting):
        delta_V[m] = sum_y[m] / sum_w[m];
    }   
    
    chi2a = chi2a / (N_data - N_PARAMS - N_filters);

    return chi2a;
}           


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
__global__ void setup_kernel ( curandState * state, unsigned long seed, CHI_FLOAT *d_f )
{
    // Global thread index:
    unsigned long long id = blockIdx.x*blockDim.x + threadIdx.x;
    // Generating initial states for all threads in a kernel:
    curand_init ( (unsigned long long)seed, id, 0, &state[id] );
    
    if (threadIdx.x==0)
        d_f[blockIdx.x] = 1e30;    

    if (threadIdx.x==0 && blockIdx.x==0)
    {
        d_block_counter = 0;
        d_sum = 0;
        d_sum2 = 0;
        d_min = 2e9;
        d_max = 0;
    }
    
    return;
} 

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__device__ int x2params(int LAM, CHI_FLOAT *x, struct parameters_struct *params, CHI_FLOAT sLimits[][N_PARAMS])
{
    double log_b, log_c; 
    
    // Checking if we went beyond the limits:
    int failed = 0;
    for (int i=0; i<N_PARAMS; i++)
    {
        // Three parameters (phi_M, phi_0, and psi_0 - for LAM=1 only, can have any value during optimization:
        if (i==1 || i==2 || i==7 && LAM)
            continue;
        
#ifdef RELAXED
        // Relaxing L and c: (physical values are enforced below)
        if (i==3 || i==4)
            continue;
#endif
        
        if (x[i]<0.0 || x[i]>=1.0)
            failed = 1;
    }
    // Crossing LAM <-> SAM is not allowed during optimization:
    if (LAM==0 && x[6]>0.5 || LAM==1 && x[6]<0.5)
        failed = 1;
    if (failed)
        return failed;
    
    int iparam = -1;
    // Independent parameters:
    iparam++;  params->theta_M =     x[iparam] * (sLimits[1][iparam]-sLimits[0][iparam]) + sLimits[0][iparam]; // 0
    iparam++;  params->phi_M =       x[iparam] * (sLimits[1][iparam]-sLimits[0][iparam]) + sLimits[0][iparam]; // 1
    iparam++;  params->phi_0 =       x[iparam] * (sLimits[1][iparam]-sLimits[0][iparam]) + sLimits[0][iparam]; // 2
    iparam++;  params->L =           x[iparam] * (sLimits[1][iparam]-sLimits[0][iparam]) + sLimits[0][iparam]; // 3
    iparam++;  log_c =               x[iparam] * (sLimits[1][iparam]-sLimits[0][iparam]) + sLimits[0][iparam]; // 4
#ifdef RELAXED
// Enforcing minimum limits on physical values of L and c:
    if (params->L<0.0 || log_c>0.0)
        return 1;
#endif
               params->c_tumb = exp(log_c);

    // Dependent parameters:
    iparam++;  log_b =              x[iparam] * log_c; // 5
               params->b_tumb = exp(log_b);
    
    // Derived values:
    double Is = (1.0+params->b_tumb*params->b_tumb) / (params->b_tumb*params->b_tumb+params->c_tumb*params->c_tumb);
    double Ii = (1.0+params->c_tumb*params->c_tumb) / (params->b_tumb*params->b_tumb+params->c_tumb*params->c_tumb);

    // Dependent parameters:    
    iparam++;  // 6  
    // Dimensionless total energy (excitation degree)
    if (LAM)
    // LAM: Es>1.0/Ii
        params->Es=2.0*(x[iparam]-0.5)*(1.0-1.0/Ii)+1.0/Ii;
    else
    // SAM: Es<1.0/Ii
        params->Es=2.0*x[iparam]*(1.0/Ii-1.0/Is)+1.0/Is;

    // Generating psi_0 (constrained by Es, Ii, Is)
    iparam++;  // 7
    double psi_min, psi_max;
    if (LAM)
    {
        psi_min = 0.0;
        psi_max = 2.0*PI;
    }
    else
    {
        psi_max = atan(sqrt(Ii*(Is-1.0/params->Es)/Is/(1.0/params->Es-Ii)));
        psi_min = -psi_max;
    }
    params->psi_0 = x[iparam]*(psi_max-psi_min) + psi_min;
    
    return 0;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__global__ void chi2_plot (struct obs_data *dData, int N_data, int N_filters,
                          struct parameters_struct *d_params, struct obs_data *dPlot, int Nplot, struct parameters_struct params)
// CUDA kernel to compute plot data from input params structure
{        
    CHI_FLOAT delta_V[MAX_FILTERS];
//    CHI_FLOAT chi2;
    
    
    // Step one: computing constants for each filter using chi^2 method
    d_chi2_plot = chi2one(params, dData, N_data, N_filters, delta_V, 0);
    
    // Step two: computing the Nplots data points using the delta_V values from above:

    chi2one(params, dPlot, Nplot, N_filters, delta_V, Nplot);
    
 return;   
}



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__global__ void chi2_gpu (struct obs_data *dData, int N_data, int N_filters,
                          curandState* globalState, CHI_FLOAT *d_f, struct parameters_struct *d_params)
// CUDA kernel computing chi^2 on GPU
{        
    __shared__ struct obs_data sData[MAX_DATA];
    __shared__ CHI_FLOAT sLimits[2][N_PARAMS];
    __shared__ volatile CHI_FLOAT s_f[BSIZE];
    __shared__ volatile int s_thread_id[BSIZE];
    int i, j;
    struct parameters_struct params;
    CHI_FLOAT delta_V[MAX_FILTERS];
    
    // Not efficient, for now:
    if (threadIdx.x == 0)
    {
        for (i=0; i<N_data; i++)
            sData[i] = dData[i];
        for (i=0; i<N_INDEPEND; i++)
        {
            sLimits[0][i] = dLimits[0][i];
            sLimits[1][i] = dLimits[1][i];
        }
    }
    
    // Downhill simplex optimization approach
    
    __syncthreads();
    
    // Global thread index:
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    // Reading the global states from device memory:
    curandState localState = globalState[id];
    
    int l = 0;
    
    CHI_FLOAT x[N_PARAMS+1][N_PARAMS];  // simplex points (point index, coordinate)
    CHI_FLOAT f[N_PARAMS+1]; // chi2 values for the simplex edges (point index)
    int ind[N_PARAMS+1]; // Indexes to the sorted array (point index)
    
    int LAM;
    // Initial random point
    for (i=0; i<N_PARAMS; i++)
    {
        float r = curand_uniform(&localState);
        if (i == 6)
        {
            // Using the x value for Es to determine the mode (1:LAM. 0:SAM)
            LAM = r>=0.5;
            if (LAM==0)
                // Interval 1e-6 ... 0.5-DX_INI-1e-6:
                x[0][i] = 1e-6 + (1-2*DX_INI-4e-6) * r;
            else
                // Interval 0.5+1e-6 ... 1-DX_INI-1e-6:
                x[0][i] = 0.5 + 1e-6 + (1-2*DX_INI-4e-6) * (r-0.5);
        }
        else
        // The DX_INI business is to prevent the initial simplex going beyong the limits (???)
            x[0][i] = 1e-6 + (1.0-DX_INI-2e-6) * r;
    }
    
    // Simplex initialization
    for (j=1; j<N_PARAMS+1; j++)
    {
        for (i=0; i<N_PARAMS; i++)
        {
            if (i == j-1)
                x[j][i] = x[0][i] + DX_INI;
            else
                x[j][i] = x[0][i];
        }
    }
    
    // Computing the initial function values (chi2):        
    for (j=0; j<N_PARAMS+1; j++)
    {
        x2params(LAM, x[j], &params, sLimits);
        f[j] = chi2one(params, sData, N_data, N_filters, delta_V, 0);    
    }
        
    // The main simplex loop
    while (1)
    {
        l++;  // Incrementing the global (for the whole lifetime of the thread) simplex steps counter by one
        
        // Sorting the simplex:
        bool ind2[N_PARAMS+1];
        for (j=0; j<N_PARAMS+1; j++)
        {
            ind2[j] = 0;  // Uninitialized flag
        }
        CHI_FLOAT fmin;
        int jmin, j2;
        for (j=0; j<N_PARAMS+1; j++)
        {
            fmin = 1e30;
            for (j2=0; j2<N_PARAMS+1; j2++)
            {
                if (ind2[j2]==0 && f[j2] <= fmin)
                {
                    fmin = f[j2];
                    jmin = j2;
                }            
            }
            ind[j] = jmin;
            ind2[jmin] = 1;
        }    
        
        // Simplex centroid:
        CHI_FLOAT x0[N_PARAMS];
        for (i=0; i<N_PARAMS; i++)
        {
            CHI_FLOAT sum = 0.0;
            for (j=0; j<N_PARAMS+1; j++)
                sum = sum + x[j][i];
            x0[i] = sum / (N_PARAMS+1);
        }           
        
        // Simplex size squared:
        CHI_FLOAT size2 = 0.0;
        for (j=0; j<N_PARAMS+1; j++)
        {
            CHI_FLOAT sum = 0.0;
            for (i=0; i<N_PARAMS; i++)
            {
                CHI_FLOAT dx = x[j][i] - x0[i];
                sum = sum + dx*dx;
            }
            size2 = size2 + sum;
        }
        size2 = size2 / N_PARAMS;  // Computing the std square of the simplex points relative to the centroid point
        
        // Simplex convergence criterion, plus the end of thread life criterion:
        /*
         *            if (size2 < SIZE2_MIN || l-l0>NS_STEPS || l > N_STEPS)
         *            {
         *                l0 = l;
         *                break;
    }
    */
        if (size2 < SIZE2_MIN)
            // We converged
            break;
        if (l > N_STEPS)
            // We ran out of time
            break;
        
        // Reflection
        CHI_FLOAT x_r[N_PARAMS];
        for (i=0; i<N_PARAMS; i++)
        {
            x_r[i] = x0[i] + ALPHA_SIM*(x0[i] - x[ind[N_PARAMS]][i]);
        }
        CHI_FLOAT f_r;
        if (x2params(LAM, x_r,&params,sLimits))
            f_r = 1e30;
        else
            f_r = chi2one(params, sData, N_data, N_filters, delta_V, 0);
        if (f_r >= f[ind[0]] && f_r < f[ind[N_PARAMS-1]])
        {
            // Replacing the worst point with the reflected point:
            for (i=0; i<N_PARAMS; i++)
            {
                x[ind[N_PARAMS]][i] = x_r[i];
            }
            f[ind[N_PARAMS]] = f_r;
            continue;  // Going to the next simplex step
        }
        
        // Expansion
        if (f_r < f[ind[0]])
        {
            CHI_FLOAT x_e[N_PARAMS];
            for (i=0; i<N_PARAMS; i++)
            {
                x_e[i] = x0[i] + GAMMA_SIM*(x_r[i] - x0[i]);
            }
            CHI_FLOAT f_e;
            if (x2params(LAM, x_e,&params,sLimits))
                f_e = 1e30;
            else
                f_e = chi2one(params, sData, N_data, N_filters, delta_V, 0);
            if (f_e < f_r)
            {
                // Replacing the worst point with the expanded point:
                for (i=0; i<N_PARAMS; i++)
                {
                    x[ind[N_PARAMS]][i] = x_e[i];
                }
                f[ind[N_PARAMS]] = f_e;
            }
            else
            {
                // Replacing the worst point with the reflected point:
                for (i=0; i<N_PARAMS; i++)
                {
                    x[ind[N_PARAMS]][i] = x_r[i];
                }
                f[ind[N_PARAMS]] = f_r;
            }
            continue;  // Going to the next simplex step
        }
        
        // Contraction
        // (Here we repurpose x_r and f_r for the contraction stuff)
        for (i=0; i<N_PARAMS; i++)
        {
            x_r[i] = x0[i] + RHO_SIM*(x[ind[N_PARAMS]][i] - x0[i]);
        }
        if (x2params(LAM, x_r,&params,sLimits))
            f_r = 1e30;
        else
            f_r = chi2one(params, sData, N_data, N_filters, delta_V, 0);
        if (f_r < f[ind[N_PARAMS]])
        {
            // Replacing the worst point with the contracted point:
            for (i=0; i<N_PARAMS; i++)
            {
                x[ind[N_PARAMS]][i] = x_r[i];
            }
            f[ind[N_PARAMS]] = f_r;
            continue;  // Going to the next simplex step
        }
        
        // If all else fails - shrink
        bool failed = 0;
        for (j=1; j<N_PARAMS+1; j++)
        {
            for (i=0; i<N_PARAMS; i++)
            {
                x[ind[j]][i] = x[ind[0]][i] + SIGMA_SIM*(x[ind[j]][i] - x[ind[0]][i]);
            }           
            if (x2params(LAM, x[ind[j]],&params,sLimits))
                failed = 1;
            else
                f[ind[j]] = chi2one(params, sData, N_data, N_filters, delta_V, 0);
        }
        // We failed the optimization
        if (failed)
        {
            f[ind[0]] = 1e30;
            break;
        }
        
    }  // inner while loop
    
    
    s_f[threadIdx.x] = f[ind[0]];
    s_thread_id[threadIdx.x] = threadIdx.x;
    
    __syncthreads();
    //!!! AT this point not all warps initialized s_f and s_thread_id! It looks like __syncthreads doesn't work!
    
    // Binary reduction:
    int nTotalThreads = blockDim.x;
    while(nTotalThreads > 1)
    {
        int halfPoint = nTotalThreads / 2; // Number of active threads
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint; // the second element index
            float temp = s_f[thread2];
            if (temp < s_f[threadIdx.x])
            {
                s_f[threadIdx.x] = temp;
                s_thread_id[threadIdx.x] = s_thread_id[thread2];
            }
        }
        __syncthreads();
        nTotalThreads = halfPoint; // Reducing the binary tree size by two
    }
    // At this point, the smallest chi2 in the block is in s_f[0]
    
    if (threadIdx.x == s_thread_id[0])
    {
        unsigned int blockID = atomicAdd(&d_block_counter, 1);
        // Copying the found minimum to device memory:
        d_f[blockID] = s_f[0];
        x2params(LAM, x[ind[0]],&params,sLimits);
        d_params[blockID] = params;
    }
    
    // Very expensive: probably should only be used for debugging:
    atomicAdd(&d_sum, (unsigned long long)l);
    atomicAdd(&d_sum2, (unsigned long long)l*(unsigned long long)l);
    atomicMin(&d_min, l);
    atomicMax(&d_max, l);
        
    return;        
    
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// For debugging:
#ifdef DEBUG
__global__ void debug_kernel(struct parameters_struct params, struct obs_data *dData, int N_data, int N_filters)
{
    __shared__ struct obs_data sData[MAX_DATA];
    int i;
    CHI_FLOAT f;
    
    // Not efficient, for now:
    if (threadIdx.x == 0)
    {
        for (i=0; i<N_data; i++)
            sData[i] = dData[i];
    }

f = chi2one(params, sData, N_data, N_filters, delta_V, 0);
    
return;

}

#endif


/*            
            // Initial angles values = the old values, from the previous i cycle:            
            // RK4 method for solving ODEs with a fixed time step h
            for (int l=0; l<N_steps; l++)
            {
                double y[3], K[3], f[3];
                
                y[0] = phi;
                y[1] = theta;
                y[2] = psi;
                ODE_func (y, K, mu);
                f[0] = phi + h/6.0 *K[0];
                f[1] = theta + h/6.0 *K[1];
                f[2] = psi + h/6.0 *K[2];
        
                y[0] = phi + 0.5*h*K[0];
                y[1] = theta + 0.5*h*K[1];
                y[2] = psi + 0.5*h*K[2];
                ODE_func (y, K, mu);
                f[0] = f[0] + h/3.0 *K[0];
                f[1] = f[1] + h/3.0 *K[1];
                f[2] = f[2] + h/3.0 *K[2];
        
                y[0] = phi + 0.5*h*K[0];
                y[1] = theta + 0.5*h*K[1];
                y[2] = psi + 0.5*h*K[2];
                ODE_func (y, K, mu);
                f[0] = f[0] + h/3.0 *K[0];
                f[1] = f[1] + h/3.0 *K[1];
                f[2] = f[2] + h/3.0 *K[2];

                y[0] = phi + h*K[0];
                y[1] = theta + h*K[1];
                y[2] = psi + h*K[2];
                ODE_func (y, K, mu);
                phi = f[0] + h/6.0 *K[0];
                theta = f[1] + h/6.0 *K[1];
                psi = f[2] + h/6.0 *K[2];
        
            }
            */
