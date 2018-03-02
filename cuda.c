/* CUDA stuff.
 * Computing chi^2 on GPU for a given combination of free parameters
 * 
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include "asteroid.h"


__device__ CHI_FLOAT chi2one(struct parameters_struct params, struct obs_data *sData, int N_data, int N_filters)
// Computung chi^2 for a single model parameters combination, on GPU, by a single thread
{
    int i, m;
    double phi_a;
    double n_x, n_y, n_z;
    double cos_phi_a, sin_phi_a, cos_alpha_p, sin_alpha_p, scalar_Sun, scalar_Earth, scalar;
    double cos_lambda_p, sin_lambda_p, Vmod, alpha_p, lambda_p;
    double a_x,a_y,a_z,b_x,b_y,b_z,c_x,c_y,c_z;
    double Ep_x, Ep_y, Ep_z, Sp_x, Sp_y, Sp_z;
    CHI_FLOAT chi2a;
    double sum_y2[N_FILTERS];
    double sum_y[N_FILTERS];
    double sum_w[N_FILTERS];
    
    for (m=0; m<N_filters; m++)
    {
        sum_y2[m] = 0.0;
        sum_y[m] = 0.0;
        sum_w[m] = 0.0;
    }
    
    // Calculations which are time independent:
    
    #ifdef TUMBLE    
    // In tumbling mode, the fixed vector pr is the precession vector
    double pr_phi = acos(params.cos_phi);
    double pr_x = sin(params.theta)*params.cos_phi;
    double pr_y = sin(params.theta)*sin(pr_phi);
    double pr_z = cos(params.theta);
    
    double cos_theta_pr = cos(params.theta_pr);
    double sin_theta_pr = sin(params.theta_pr);
    
    // Initial (phase=0) vector n orientation; it is in pr-0-pp plane, where pp=[z x pr], made a unit vector
    double n0_x = pr_x*cos_theta_pr - pr_y/sqrt(pr_y*pr_y+pr_x*pr_x)*sin_theta_pr;
    double n0_y = pr_y*cos_theta_pr + pr_x/sqrt(pr_y*pr_y+pr_x*pr_x)*sin_theta_pr;
    double n0_z = pr_z*cos_theta_pr;
    #endif    
    
    // The loop over all data points    
    for (i=0; i<N_data; i++)
    {            
        
        #ifdef TUMBLE    
        // Dot product:
        double pr_n0 = pr_x*n0_x + pr_y*n0_y + pr_z*n0_z;
        double phi_n = params.phi_n0 + sData[i].MJD/params.P_pr * 2*PI;
        double cos_phi_n = cos(phi_n);
        double sin_phi_n = sin(phi_n);        
        // Using the Rodrigues formula to rotate the internal spin vector n around the precession vector pr by angle phi_n:
        n_x = n0_x*cos_phi_n + (pr_y*n0_z - pr_z*n0_y)*sin_phi_n + pr_x*pr_n0*(1.0-cos_phi_n);
        n_y = n0_y*cos_phi_n + (pr_z*n0_x - pr_x*n0_z)*sin_phi_n + pr_y*pr_n0*(1.0-cos_phi_n);
        n_z = n0_z*cos_phi_n + (pr_x*n0_y - pr_y*n0_x)*sin_phi_n + pr_z*pr_n0*(1.0-cos_phi_n);        
        #else    
        // Spin vector (barycentric FoR); https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
        double n_phi = acos(params.cos_phi);
        n_x = sin(params.theta)*params.cos_phi;
        n_y = sin(params.theta)*sin(n_phi);
        n_z = cos(params.theta);
        #endif                
        
        double a0_x, a0_y, a0_z;
        
        // Initial (phase=0) vector a0 orientation; it is in n-0-p plane, where p=[z x n], made a unit vector
        a0_x = - n_y/sqrt(n_y*n_y+n_x*n_x);
        a0_y =   n_x/sqrt(n_y*n_y+n_x*n_x);
        a0_z =   0.0;
        
        // Vector b_i (axis b before applying the phi_b rotation), vector product [a_0 x n]:
        double bi_x = a0_y*n_z - a0_z*n_y;
        double bi_y = a0_z*n_x - a0_x*n_z;
        double bi_z = a0_x*n_y - a0_y*n_x;
        // Making it a unit vector:
        double bi = sqrt(bi_x*bi_x + bi_y*bi_y + bi_z*bi_z);
        bi_x = bi_x / bi;
        bi_y = bi_y / bi;
        bi_z = bi_z / bi;
        
        // Vector t=[a0 x bi]:
        double t_x = a0_y*bi_z - a0_z*bi_y;
        double t_y = a0_z*bi_x - a0_x*bi_z;
        double t_z = a0_x*bi_y - a0_y*bi_x;
        // Making it a unit vector:
        double t = sqrt(t_x*t_x + t_y*t_y + t_z*t_z);
        t_x = t_x / t;
        t_y = t_y / t;
        t_z = t_z / t;
        
        // Initial (phase=0) axis b0:
        double phi_b = acos(params.cos_phi_b);
        double sin_phi_b = sin(phi_b);
        double b0_x = bi_x*params.cos_phi_b + t_x*sin_phi_b;
        double b0_y = bi_y*params.cos_phi_b + t_y*sin_phi_b;
        double b0_z = bi_z*params.cos_phi_b + t_z*sin_phi_b;
        
        // Dot products:
        double n_a0 = n_x*a0_x + n_y*a0_y + n_z*a0_z;
        double n_b0 = n_x*b0_x + n_y*b0_y + n_z*b0_z;                                        
        
        phi_a = params.phi_a0 + sData[i].MJD/params.P * 2*PI;
        
        cos_phi_a = cos(phi_a);
        sin_phi_a = sin(phi_a);
        
        // New basis - a,b,c axes of the ellipsoid after the phase rotation:
        // Using the Rodrigues formula for a and b axes (n is the axis of rotation vector; a0 is the initial vector; a is the vector after rotation of phi_a radians)
        // https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        a_x = a0_x*cos_phi_a + (n_y*a0_z - n_z*a0_y)*sin_phi_a + n_x*n_a0*(1.0-cos_phi_a);
        a_y = a0_y*cos_phi_a + (n_z*a0_x - n_x*a0_z)*sin_phi_a + n_y*n_a0*(1.0-cos_phi_a);
        a_z = a0_z*cos_phi_a + (n_x*a0_y - n_y*a0_x)*sin_phi_a + n_z*n_a0*(1.0-cos_phi_a);
        
        b_x = b0_x*cos_phi_a + (n_y*b0_z - n_z*b0_y)*sin_phi_a + n_x*n_b0*(1.0-cos_phi_a);
        b_y = b0_y*cos_phi_a + (n_z*b0_x - n_x*b0_z)*sin_phi_a + n_y*n_b0*(1.0-cos_phi_a);
        b_z = b0_z*cos_phi_a + (n_x*b0_y - n_y*b0_x)*sin_phi_a + n_z*n_b0*(1.0-cos_phi_a);
        
        // c = [a x b]:
        c_x = a_y*b_z - a_z*b_y;
        c_y = a_z*b_x - a_x*b_z;
        c_z = a_x*b_y - a_y*b_x;
        
        // Earth vector in the new (a,b,c) basis:
        Ep_x = a_x*sData[i].E_x + a_y*sData[i].E_y + a_z*sData[i].E_z;
        Ep_y = b_x*sData[i].E_x + b_y*sData[i].E_y + b_z*sData[i].E_z;
        Ep_z = c_x*sData[i].E_x + c_y*sData[i].E_y + c_z*sData[i].E_z;
        
        // Sun vector in the new (a,b,c) basis:
        Sp_x = a_x*sData[i].S_x + a_y*sData[i].S_y + a_z*sData[i].S_z;
        Sp_y = b_x*sData[i].S_x + b_y*sData[i].S_y + b_z*sData[i].S_z;
        Sp_z = c_x*sData[i].S_x + c_y*sData[i].S_y + c_z*sData[i].S_z;
        
        // Now that we converted the Earth and Sun vectors to the internal asteroidal basis (a,b,c),
        // we can apply the formalism of Muinonen & Lumme, 2015 to calculate the brightness of the asteroid.
        
        // The two scalars from eq.(12) of Muinonen & Lumme, 2015:
        scalar_Sun = sqrt(Sp_x*Sp_x + Sp_y*Sp_y/(params.b*params.b) + Sp_z*Sp_z/(params.c*params.c));
        scalar_Earth = sqrt(Ep_x*Ep_x + Ep_y*Ep_y/(params.b*params.b) + Ep_z*Ep_z/(params.c*params.c));
        
        // From eq.(13):
        cos_alpha_p = (Sp_x*Ep_x + Sp_y*Ep_y/(params.b*params.b) + Sp_z*Ep_z/(params.c*params.c)) / (scalar_Sun * scalar_Earth);
        sin_alpha_p = sqrt(1.0 - cos_alpha_p*cos_alpha_p);
        alpha_p = atan2(sin_alpha_p, cos_alpha_p);
        
        // From eq.(14):
        scalar = sqrt(scalar_Sun*scalar_Sun + scalar_Earth*scalar_Earth + 2*scalar_Sun*scalar_Earth*cos_alpha_p);
        cos_lambda_p = (scalar_Sun + scalar_Earth*cos_alpha_p) / scalar;
        sin_lambda_p = scalar_Earth*sin_alpha_p / scalar;
        lambda_p = atan2(sin_lambda_p, cos_lambda_p);
        
        // Asteroid's model visual brightness, from eq.(10):
        Vmod = -2.5*log10(params.b*params.c * scalar_Sun*scalar_Earth/scalar * (cos(lambda_p-alpha_p) + cos_lambda_p +
        sin_lambda_p*sin(lambda_p-alpha_p) * log(1.0 / tan(0.5*lambda_p) / tan(0.5*(alpha_p-lambda_p)))));
        
        
        // Filter:
        int m = sData[i].Filter;
        // Difference between the observational and model magnitudes:
        double y = sData[i].V - Vmod;                    
        sum_y2[m] = sum_y2[m] + y*y*sData[i].w;
        sum_y[m] = sum_y[m] + y*sData[i].w;
        sum_w[m] = sum_w[m] + sData[i].w;
        
    } // data points loop
    
    
    CHI_FLOAT chi2m;
    chi2a=0.0;    
    for (m=0; m<N_filters; m++)
    {
        // Chi^2 for the m-th filter:
        chi2m = sum_y2[m] - sum_y[m]*sum_y[m]/sum_w[m];
        chi2a = chi2a + chi2m;
    }   
    
    chi2a = chi2a / (N_data - N_PARAMS - N_filters);
    return chi2a;
}           


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ifdef SIMPLEX
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

__device__ int x2params(CHI_FLOAT *x, struct parameters_struct *params)
{
    // Checking if we went beyond the limits:
    int failed = 0;
    for (int i=0; i<N_PARAMS; i++)
    {
        if (x[i]<=0.0 || x[i]>=1.0)
            failed = 1;
    }
    if (failed)
        return failed;
    
    params->b =       x[0] * (dLimits[1][0]-dLimits[0][0]) + dLimits[0][0];
    params->P =       x[1] * (dLimits[1][1]-dLimits[0][1]) + dLimits[0][1];
    params->theta =   x[2] * (dLimits[1][2]-dLimits[0][2]) + dLimits[0][2]; 
    params->cos_phi = x[3] * (dLimits[1][3]-dLimits[0][3]) + dLimits[0][3];
    params->phi_a0 =  x[4] * (dLimits[1][4]-dLimits[0][4]) + dLimits[0][4];
    // &&&    
    params->c =       x[5] * (dLimits[1][5]-dLimits[0][5]) + dLimits[0][5];
    params->cos_phi_b=x[6] * (dLimits[1][6]-dLimits[0][6]) + dLimits[0][6];
    #ifdef TUMBLE
    params->P_pr    = x[7] * (dLimits[1][7]-dLimits[0][7]) + dLimits[0][7]; 
    params->theta_pr =x[8] * (dLimits[1][8]-dLimits[0][8]) + dLimits[0][8]; 
    params->phi_n0  = x[9] * (dLimits[1][9]-dLimits[0][9]) + dLimits[0][9]; 
    #endif    
    
    #ifdef FORCE_BC
    if (params->c > params->b)
        return 1;
    #endif    
    #ifdef LOG_BC
    params->b = exp(params->b);
    params->c = exp(params->c);
    #endif    
    
    return 0;
}
#endif



//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifdef SIMPLEX
__global__ void chi2_gpu (struct obs_data *dData, int N_data, int N_filters,
                          curandState* globalState, CHI_FLOAT *d_f, struct parameters_struct *d_params)
#else
__global__ void chi2_gpu (struct obs_data *dData, int N_data, int N_filters, long int Nloc, int iglob, int N_serial, float * d_chi2_min, long int * d_iloc_min)
#endif
// CUDA kernel computing chi^2 on GPU
{        
    __shared__ struct obs_data sData[MAX_DATA];
    int i, j;
    struct parameters_struct params;
    
    // Not efficient, for now:
    if (threadIdx.x == 0)
    {
        for (i=0; i<N_data; i++)
            sData[i] = dData[i];
    }
    
    #ifdef SIMPLEX
    // Downhill simplex optimization approach
    __shared__ CHI_FLOAT s_f[BSIZE];
    __shared__ int s_thread_id[BSIZE];
    
    __syncthreads();
    
    // Global thread index:
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    // Reading the global states from device memory:
    curandState localState = globalState[id];
    
    int l = 0;
    
    CHI_FLOAT x[N_PARAMS+1][N_PARAMS];  // simplex points (point index, coordinate)
    CHI_FLOAT f[N_PARAMS+1]; // chi2 values for the simplex edges (point index)
    int ind[N_PARAMS+1]; // Indexes to the sorted array (point index)
    
    
    // Initial random point
    for (i=0; i<N_PARAMS; i++)
    {
        // The DX_INI business is to prevent the initial simplex going beyong the limits
        x[0][i] = 1e-6 + (1.0-DX_INI-2e-6)*curand_uniform(&localState);
    }
    #ifdef FORCE_BC
    // Enforcing c<b initially (not perfect - for very small b might fail at the beginning)
    x[0][5] = x[0][5] * x[0][0];
    #endif                    
    
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
        x2params(x[j],&params);
        f[j] = chi2one(params, sData, N_data, N_filters);    
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
        if (x2params(x_r,&params))
            f_r = 1e30;
        else
            f_r = chi2one(params, sData, N_data, N_filters);
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
            if (x2params(x_e,&params))
                f_e = 1e30;
            else
                f_e = chi2one(params, sData, N_data, N_filters);
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
        if (x2params(x_r,&params))
            f_r = 1e30;
        else
            f_r = chi2one(params, sData, N_data, N_filters);
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
            if (x2params(x[ind[j]],&params))
                failed = 1;
            else
                f[ind[j]] = chi2one(params, sData, N_data, N_filters);
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
        x2params(x[ind[0]],&params);
        d_params[blockID] = params;
    }
    
    // Very expensive: probably should only be used for debugging:
    atomicAdd(&d_sum, (unsigned long long)l);
    atomicAdd(&d_sum2, (unsigned long long)l*(unsigned long long)l);
    atomicMin(&d_min, l);
    atomicMax(&d_max, l);
        
    return;
    
    
    #else
    // Brute force approach (exploring the whole parameter space)
    
    __shared__ float s_chi2_min[BSIZE];
    __shared__ int s_thread_id[BSIZE];
    float chi2a, chi2_min;
    long int iloc, iloc_min;
    
    chi2_min = 1e30;
    
    __syncthreads();
    
    // Global index (for the parameters for which we compute separate Chi^2_min)
    // It only changes with blockIds
    //    iglob = blockIdx.y + gridDim.y*blockIdx.z;
    iglob_to_params(&iglob, &params);
    
    // Local index (for parameters over which chi2_min is computed)
    // Changes with blockId.x, thread ID, and serial loop index j
    iloc = N_serial * ((long int)threadIdx.x + blockDim.x*(long int)blockIdx.x);
    if (iloc < Nloc)
    {
        
        
        // Outer serial loop:
        for (j=0; j<N_serial; j++)
        {
            
            iloc_to_params(&iloc, &params);
            
            chi2a = chi2one(params, sData, N_data, N_filters);
            
            if (chi2a < chi2_min)
            {
                chi2_min = chi2a;
                iloc_min = iloc;
            }                
            
            iloc++;
            if (iloc >= Nloc)
                break;
            
        } // j loop (serial)            
    }
    
    s_chi2_min[threadIdx.x] = chi2_min;
    s_thread_id[threadIdx.x] = threadIdx.x;
    
    __syncthreads();
    
    // Binary reduction:
    int nTotalThreads = blockDim.x;
    while(nTotalThreads > 1)
    {
        int halfPoint = nTotalThreads / 2; // Number of active threads
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint; // the second element index
            float temp = s_chi2_min[thread2];
            if (temp < s_chi2_min[threadIdx.x])
            {
                s_chi2_min[threadIdx.x] = temp;
                s_thread_id[threadIdx.x] = s_thread_id[thread2];
            }
        }
        __syncthreads();
        nTotalThreads = halfPoint; // Reducing the binary tree size by two
    }
    // At this point, the smallest chi2 in the block is in s_chi2_min[0]
    
    if (threadIdx.x == 0)
    {
        // Copying the found minimum to device memory:
        d_chi2_min[blockIdx.x] = s_chi2_min[0];
    }
    if (threadIdx.x == s_thread_id[0])
    {
        // Copying the found minimum to device memory:
        d_iloc_min[blockIdx.x] = iloc_min;
    }
    
    return;
    
    #endif
    
    
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
