/* CUDA stuff.
 * Computing chi^2 on GPU for a given combination of free parameters
 * 
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "asteroid.h"

__global__ void chi2_gpu (struct obs_data *dData, int N_data, int N_filters)
{        
    __shared__ struct obs_data sData;
    
    int i, m, l, N;
    double phi_a, cos_n_phi, n_theta, theta_a, cos_phi_b, b, c, P;
    double n_phi, n_x, n_y, n_z;
    double cos_phi_a, sin_phi_a, cos_alpha_p, sin_alpha_p, scalar_Sun, scalar_Earth, scalar;
    double cos_lambda_p, sin_lambda_p, Vmod, alpha_p, lambda_p;
    double a_x,a_y,a_z,b_x,b_y,b_z,c_x,c_y,c_z;
    double Ep_x, Ep_y, Ep_z, Sp_x, Sp_y, Sp_z;
    double chi2a, chi2_min;
    
    double * sum_y2 = (double *)malloc(N_filters*sizeof(double));
    double * sum_y = (double *)malloc(N_filters*sizeof(double));
    double * sum_w = (double *)malloc(N_filters*sizeof(double));
    
    chi2_min = 1e30;
    
    // Free parameters:
    b = 0.2;
    c = 0.2;
    cos_n_phi = cos(10/RAD);
    n_theta = 0.001 / RAD;
    theta_a = 90 / RAD;
    cos_phi_b = 0.0;
    //    P = 7.35 / 24.0; // 7.34
    // P interval (days):
    const double P1 = 7.4 / 24.0;
    const double P2 = 7.5 /24.0;
    // Number of points for P (period):
    const int N_P = 100;
    
    
    // Disk: 1, 0.165, 5, 60, 90, 0.03, 7.35: 12.11:
    // Disk P/2: 1.2, 0.05, 4.5, 69, 90, 0.52, 3.66: 14.65
    // Cigar: 0.22, 0.21, 10, 0.001, 90, 0, 7.33
    
    // Disk for dat3: 1.01, 0.155, 4, 63, 90, 0.03, 7.353
    
    // Calculations which are time independent:
    
    // Spin vector (barycentric FoR); https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
    n_phi = acos(cos_n_phi);
    n_x = sin(n_theta)*cos_n_phi;
    n_y = sin(n_theta)*sin(n_phi);
    n_z = cos(n_theta);
    
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
    double phi_b = acos(cos_phi_b);
    double sin_phi_b = sin(phi_b);
    double b0_x = bi_x*cos_phi_b + t_x*sin_phi_b;
    double b0_y = bi_y*cos_phi_b + t_y*sin_phi_b;
    double b0_z = bi_z*cos_phi_b + t_z*sin_phi_b;
    
    // Dot products:
    double n_a0 = n_x*a0_x + n_y*a0_y + n_z*a0_z;
    double n_b0 = n_x*b0_x + n_y*b0_y + n_z*b0_z;
    
    int i_phi_a, i_phi_a_min = 0;
    int i_P, i_P_min = 0;
    double phi_a0;
    
    
    // Loop over periods
    for (i_P=0; i_P<N_P; i_P++)
    {
        P = (double)i_P/(double)N_P * (P2 - P1) + P1;
        
        // The loop over phase shifts:
        for (i_phi_a=0; i_phi_a<N_PHI_A; i_phi_a++)
        {
            for (m=0; m<N_filters; m++)
            {
                sum_y2[m] = 0.0;
                sum_y[m] = 0.0;
                sum_w[m] = 0.0;
            }
            
            phi_a0 = (double)i_phi_a/(double)N_PHI_A * 2*PI;
            
            // The loop over all data points    
            for (i=0; i<N; i++)
            {            
                // The expensive step - copying observational data for this momemt of time from device to shared memory:
                sData = dData[i];
                
                phi_a = phi_a0 + sData.MJD/P * 2*PI;
                                
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
                Ep_x = a_x*sData.E_x + a_y*sData.E_y + a_z*sData.E_z;
                Ep_y = b_x*sData.E_x + b_y*sData.E_y + b_z*sData.E_z;
                Ep_z = c_x*sData.E_x + c_y*sData.E_y + c_z*sData.E_z;
                
                // Sun vector in the new (a,b,c) basis:
                Sp_x = a_x*sData.S_x + a_y*sData.S_y + a_z*sData.S_z;
                Sp_y = b_x*sData.S_x + b_y*sData.S_y + b_z*sData.S_z;
                Sp_z = c_x*sData.S_x + c_y*sData.S_y + c_z*sData.S_z;
                
                // Now that we converted the Earth and Sun vectors to the internal asteroidal basis (a,b,c),
                // we can apply the formalism of Muinonen & Lumme, 2015 to calculate the brightness of the asteroid.
                
                // The two scalars from eq.(12) of Muinonen & Lumme, 2015:
                scalar_Sun = sqrt(Sp_x*Sp_x + Sp_y*Sp_y/(b*b) + Sp_z*Sp_z/(c*c));
                scalar_Earth = sqrt(Ep_x*Ep_x + Ep_y*Ep_y/(b*b) + Ep_z*Ep_z/(c*c));
                
                // From eq.(13):
                cos_alpha_p = (Sp_x*Ep_x + Sp_y*Ep_y/(b*b) + Sp_z*Ep_z/(c*c)) / (scalar_Sun * scalar_Earth);
                sin_alpha_p = sqrt(1.0 - cos_alpha_p*cos_alpha_p);
                alpha_p = atan2(sin_alpha_p, cos_alpha_p);
                
                // From eq.(14):
                scalar = sqrt(scalar_Sun*scalar_Sun + scalar_Earth*scalar_Earth + 2*scalar_Sun*scalar_Earth*cos_alpha_p);
                cos_lambda_p = (scalar_Sun + scalar_Earth*cos_alpha_p) / scalar;
                sin_lambda_p = scalar_Earth*sin_alpha_p / scalar;
                lambda_p = atan2(sin_lambda_p, cos_lambda_p);
                
                // Asteroid's model visual brightness, from eq.(10):
                Vmod = -2.5*log10(b*c * scalar_Sun*scalar_Earth/scalar * (cos(lambda_p-alpha_p) + cos_lambda_p +
                sin_lambda_p*sin(lambda_p-alpha_p) * log(1.0 / tan(0.5*lambda_p) / tan(0.5*(alpha_p-lambda_p)))));
                
                
                // Filter:
                int m = sData.Filter;
                // Difference between the observational and model magnitudes:
                double y = sData.V - Vmod;                    
                sum_y2[m] = sum_y2[m] + y*y*sData.w;
                sum_y[m] = sum_y[m] + y*sData.w;
                sum_w[m] = sum_w[m] + sData.w;
                
                
            } // data points loop
            
            
            double chi2m;
            chi2a=0.0;    
            for (m=0; m<N_filters; m++)
            {
                // Chi^2 for the m-th filter:
                chi2m = sum_y2[m] - sum_y[m]*sum_y[m]/sum_w[m];
                chi2a = chi2a + chi2m;
            }   
            
            //                chi2a = chi2a / (N_data - N_PARAMS - N_filters);
            if (chi2a < chi2_min)
            {
                chi2_min = chi2a;
                i_phi_a_min = i_phi_a;
                i_P_min = i_P;
            }
            
            
        } // i_phi_a loop
    } // i_P loop
}
