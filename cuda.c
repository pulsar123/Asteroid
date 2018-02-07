/* CUDA stuff.
 * Computing chi^2 on GPU for a given combination of free parameters
 * 
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include "asteroid.h"



__global__ void chi2_gpu (struct obs_data *dData, int N_data, int N_filters, long int Nloc, int iglob, int N_serial, double * d_chi2_min, long int * d_iloc_min)
{        
    __shared__ struct obs_data sData[N_WARPS];
    __shared__ double s_chi2_min[BSIZE];
    __shared__ long int s_iloc_min[BSIZE];
    
    int i, m, j;
    double theta_a, n_phi, phi_a;
    double n_x, n_y, n_z;
    double cos_phi_a, sin_phi_a, cos_alpha_p, sin_alpha_p, scalar_Sun, scalar_Earth, scalar;
    double cos_lambda_p, sin_lambda_p, Vmod, alpha_p, lambda_p;
    double a_x,a_y,a_z,b_x,b_y,b_z,c_x,c_y,c_z;
    double Ep_x, Ep_y, Ep_z, Sp_x, Sp_y, Sp_z;
    double chi2a, chi2_min;
    struct parameters_struct params;
    //    int iglob;
    long int iloc, iloc_min;
    
    double * sum_y2 = (double *)malloc(N_filters*sizeof(double));
    double * sum_y = (double *)malloc(N_filters*sizeof(double));
    double * sum_w = (double *)malloc(N_filters*sizeof(double));
    
    
    chi2_min = 1e30;
    
    // Global index (for the parameters for which we compute separate Chi^2_min)
    // It only changes with blockIds
    //    iglob = blockIdx.y + gridDim.y*blockIdx.z;
    iglob_to_params(&iglob, &params);
    
    // Local index (for parameters over which chi2_min is computed)
    // Changes with blockId.x, thread ID, and serial loop index j
    iloc = N_serial * ((long int)threadIdx.x + blockDim.x*(long int)blockIdx.x);
    if (iloc < Nloc)
    {
        
        int warpID = threadIdx.x / warpSize;
        
        // Outer serial loop:
        for (j=0; j<N_serial; j++)
        {
            
            iloc_to_params(&iloc, &params);
            
            theta_a = 90.0 / RAD;    
            
            // Calculations which are time independent:
            
            // Spin vector (barycentric FoR); https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
            n_phi = acos(params.cos_phi);
            n_x = sin(params.theta)*params.cos_phi;
            n_y = sin(params.theta)*sin(n_phi);
            n_z = cos(params.theta);
            
            double cos_theta_a = cos(theta_a);
            double sin_theta_a = sin(theta_a);
            double a0_x, a0_y, a0_z;
            
            // Initial (phase=0) vector a0 orientation; it is in n-0-p plane, where p=[z x n], made a unit vector
            a0_x = n_x*cos_theta_a - n_y/sqrt(n_y*n_y+n_x*n_x)*sin_theta_a;
            a0_y = n_y*cos_theta_a + n_x/sqrt(n_y*n_y+n_x*n_x)*sin_theta_a;
            a0_z = n_z*cos_theta_a;
            
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
            
            for (m=0; m<N_filters; m++)
            {
                sum_y2[m] = 0.0;
                sum_y[m] = 0.0;
                sum_w[m] = 0.0;
            }
            
            // The loop over all data points    
            for (i=0; i<N_data; i++)
            {            
                // The expensive step - copying observational data for this momemt of time from device to shared memory:
                // As execution is only synchronous within a warp, storing the data in a warp-specific element of the shared array sData:
                sData[warpID] = dData[i];
                //__syncthreads();            
                phi_a = params.phi_a0 + sData[warpID].MJD/params.P * 2*PI;
                
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
                Ep_x = a_x*sData[warpID].E_x + a_y*sData[warpID].E_y + a_z*sData[warpID].E_z;
                Ep_y = b_x*sData[warpID].E_x + b_y*sData[warpID].E_y + b_z*sData[warpID].E_z;
                Ep_z = c_x*sData[warpID].E_x + c_y*sData[warpID].E_y + c_z*sData[warpID].E_z;
                
                // Sun vector in the new (a,b,c) basis:
                Sp_x = a_x*sData[warpID].S_x + a_y*sData[warpID].S_y + a_z*sData[warpID].S_z;
                Sp_y = b_x*sData[warpID].S_x + b_y*sData[warpID].S_y + b_z*sData[warpID].S_z;
                Sp_z = c_x*sData[warpID].S_x + c_y*sData[warpID].S_y + c_z*sData[warpID].S_z;
                
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
                int m = sData[warpID].Filter;
                // Difference between the observational and model magnitudes:
                double y = sData[warpID].V - Vmod;                    
                sum_y2[m] = sum_y2[m] + y*y*sData[warpID].w;
                sum_y[m] = sum_y[m] + y*sData[warpID].w;
                sum_w[m] = sum_w[m] + sData[warpID].w;
                
                
            } // data points loop
            
            
            double chi2m;
            chi2a=0.0;    
            for (m=0; m<N_filters; m++)
            {
                // Chi^2 for the m-th filter:
                chi2m = sum_y2[m] - sum_y[m]*sum_y[m]/sum_w[m];
                chi2a = chi2a + chi2m;
            }   
            
            chi2a = chi2a / (N_data - N_PARAMS - N_filters);
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
    s_iloc_min[threadIdx.x] = iloc_min;
    
    __syncthreads();
    
    // Binary reduction:
    int nTotalThreads = blockDim.x;
    while(nTotalThreads > 1)
    {
        int halfPoint = nTotalThreads / 2; // Number of active threads
        if (threadIdx.x < halfPoint) {
            int thread2 = threadIdx.x + halfPoint; // the second element index
            double temp = s_chi2_min[thread2];
            if (temp < s_chi2_min[threadIdx.x])
            {
                s_chi2_min[threadIdx.x] = temp;
                s_iloc_min[threadIdx.x] = s_iloc_min[thread2];
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
        d_iloc_min[blockIdx.x] = s_iloc_min[0];
    }
    
    return;
}

