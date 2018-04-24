/*  Computing chi^2 on CPU for a given combination of free parameters
 * 
 * 
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "asteroid.h"

int chi2 (int N_data, int N_filters, struct parameters_struct params, double *chi2tot)
{        
    FILE *fpd;
    FILE *fpm;
    int i, m, l, N;
    double phi_a, cos_n_phi, n_theta, b, c, P, phi_a0;
    double n_phi, n_x, n_y, n_z;
    double cos_phi_a, sin_phi_a, cos_alpha_p, sin_alpha_p, scalar_Sun, scalar_Earth, scalar;
    double cos_lambda_p, sin_lambda_p, Vmod, alpha_p, lambda_p;
    double a_x,a_y,a_z,b_x,b_y,b_z,c_x,c_y,c_z;
    double Ep_x, Ep_y, Ep_z, Sp_x, Sp_y, Sp_z;
    double chi2a, chi2_min;
    double E_x1, E_y1, E_z1, S_x1, S_y1, S_z1;
    double MJD1_obs, MJD1;
    
    struct timeval  tdr0, tdr1;
    double cputime;
    gettimeofday (&tdr0, NULL);
    
    double * sum_y2 = (double *)malloc(N_filters*sizeof(double));
    double * sum_y = (double *)malloc(N_filters*sizeof(double));
    double * sum_w = (double *)malloc(N_filters*sizeof(double));
    double * y_avr_min = (double *)malloc(N_filters*sizeof(double));
    
    chi2_min = 1e30;
    fpd = fopen("data.dat", "w");
    fpm = fopen("model.dat", "w");
    
    // Free parameters:
    b = params.b;
    P = params.P/24.0;
    c = params.c;
    n_theta = params.theta_M;
    cos_n_phi = params.phi_M;
    phi_a0 = params.phi_a0;
    
    
    // Disk: 1, 0.165, 5, 60, 90, 0.03, 7.35: 12.11:
    // Disk P/2: 1.2, 0.05, 4.5, 69, 90, 0.52, 3.66: 14.65
    // Cigar: 0.22, 0.21, 10, 0.001, 90, 0, 7.33
    
    // Disk for dat3: 1.01, 0.155, 4, 63, 90, 0.03, 7.353
    
    // Calculations which are time independent:
    
    #ifdef TUMBLE    
    // In tumbling mode, the fixed vector pr is the precession vector
    double pr_x = sin(params.theta_M)*cos(params.phi_M);
    double pr_y = sin(params.theta_M)*sin(params.phi_M);
    double pr_z = cos(params.theta_M);
    
    double cos_theta_pr = cos(params.theta_pr);
    double sin_theta_pr = sin(params.theta_pr);
    
    // Initial (phase=0) vector n orientation; it is in pr-0-pp plane, where pp=[z x pr], made a unit vector
    double n0_x = pr_x*cos_theta_pr - pr_y/sqrt(pr_y*pr_y+pr_x*pr_x)*sin_theta_pr;
    double n0_y = pr_y*cos_theta_pr + pr_x/sqrt(pr_y*pr_y+pr_x*pr_x)*sin_theta_pr;
    double n0_z = pr_z*cos_theta_pr;
    #endif    
    
    
    for (l=0; l<2; l++)
    {
        if (l == 0)
        {
            N = N_data;
        }
        else
            N = 3000;
        
        if (l == 0)
        {
            for (m=0; m<N_filters; m++)
            {
                sum_y2[m] = 0.0;
                sum_y[m] = 0.0;
                sum_w[m] = 0.0;
            }
        }
        
        
        // The loop over all data points    
        for (i=0; i<N; i++)
        {            
            
            #ifdef TUMBLE
            double pr_n0 = pr_x*n0_x + pr_y*n0_y + pr_z*n0_z;
            
            double phi_n;
            if (l == 0)
                phi_n = params.phi_n0 + hData[i].MJD/params.P_pr*24 * 2*PI;
            else
                phi_n = params.phi_n0 + (double)i/(double)N * 2*PI * hData[N_data-1].MJD/params.P_pr*24;
            
            double cos_phi_n = cos(phi_n);
            double sin_phi_n = sin(phi_n);        
            // Using the Rodrigues formula to rotate the internal spin vector n around the precession vector pr by angle phi_n:
            n_x = n0_x*cos_phi_n + (pr_y*n0_z - pr_z*n0_y)*sin_phi_n + pr_x*pr_n0*(1.0-cos_phi_n);
            n_y = n0_y*cos_phi_n + (pr_z*n0_x - pr_x*n0_z)*sin_phi_n + pr_y*pr_n0*(1.0-cos_phi_n);
            n_z = n0_z*cos_phi_n + (pr_x*n0_y - pr_y*n0_x)*sin_phi_n + pr_z*pr_n0*(1.0-cos_phi_n);        
            #else            
            // Spin vector (barycentric FoR); https://stackoverflow.com/questions/5408276/sampling-uniformly-distributed-random-points-inside-a-spherical-volume
            n_phi = acos(cos_n_phi);
            n_x = sin(n_theta)*cos_n_phi;
            n_y = sin(n_theta)*sin(n_phi);
            n_z = cos(n_theta);
            #endif    
            
            double a0_x, a0_y, a0_z;
            
            // Initial (phase=0) vector a0 orientation; it is in n-0-p plane, where p=[z x n], made a unit vector
            a0_x = - n_y/sqrt(n_y*n_y+n_x*n_x);
            a0_y =   n_x/sqrt(n_y*n_y+n_x*n_x);
            a0_z =   0.0;
            
            // Dot product:
            double n_a0 = n_x*a0_x + n_y*a0_y + n_z*a0_z;
            
            // Rotational phase angle:
            if (l == 0)
            {
                phi_a = phi_a0 + hData[i].MJD/P * 2*PI;
                
                E_x1 = hData[i].E_x;
                E_y1 = hData[i].E_y;
                E_z1 = hData[i].E_z;
                S_x1 = hData[i].S_x;
                S_y1 = hData[i].S_y;
                S_z1 = hData[i].S_z;
            }
            else
                
            {
                phi_a = phi_a0 + (double)i/(double)N * 2*PI * hData[N_data-1].MJD/P;
                MJD1 = (double)i/(double)N * hData[N_data-1].MJD + hMJD0;
                MJD1_obs = (double)i/(double)N * (MJD_obs[N_data-1] - MJD_obs[0]) + MJD_obs[0];
                quadratic_interpolation(MJD1_obs, &E_x1, &E_y1, &E_z1, &S_x1, &S_y1, &S_z1);
                double E = sqrt(E_x1*E_x1 + E_y1*E_y1+ E_z1*E_z1);
                double S = sqrt(S_x1*S_x1 + S_y1*S_y1+ S_z1*S_z1);
                // Making S,E a unit vector:
                E_x1 = E_x1 / E;
                E_y1 = E_y1 / E;
                E_z1 = E_z1 / E;
                S_x1 = S_x1 / S;
                S_y1 = S_y1 / S;
                S_z1 = S_z1 / S;                    
            }
            
            cos_phi_a = cos(phi_a);
            sin_phi_a = sin(phi_a);
            
            // New basis - a,b,c axes of the ellipsoid after the phase rotation:
            // Using the Rodrigues formula for a (n is the axis of rotation vector = -c vector; a0 is the initial vector; a is the vector after rotation of phi_a radians)
            // https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
            double a_x,a_y,a_z;
            a_x = a0_x*cos_phi_a + (n_y*a0_z - n_z*a0_y)*sin_phi_a + n_x*n_a0*(1.0-cos_phi_a);
            a_y = a0_y*cos_phi_a + (n_z*a0_x - n_x*a0_z)*sin_phi_a + n_y*n_a0*(1.0-cos_phi_a);
            a_z = a0_z*cos_phi_a + (n_x*a0_y - n_y*a0_x)*sin_phi_a + n_z*n_a0*(1.0-cos_phi_a);
            
            // Vector b =  vector product [a x n]; it's a unit vector because <a> and <n> are, and they are perpendicular to each other
            double b_x = a_y*n_z - a_z*n_y;
            double b_y = a_z*n_x - a_x*n_z;
            double b_z = a_x*n_y - a_y*n_x;
            
            // Axis <c> vector is minus <n> vector
            // Earth vector in the new (a,b,c) basis:
            Ep_x = a_x*E_x1 + a_y*E_y1 + a_z*E_z1;
            Ep_y = b_x*E_x1 + b_y*E_y1 + b_z*E_z1;
            Ep_z =-n_x*E_x1 - n_y*E_y1 - n_z*E_z1;
            
            // Sun vector in the new (a,b,c) basis:
            Sp_x = a_x*S_x1 + a_y*S_y1 + a_z*S_z1;
            Sp_y = b_x*S_x1 + b_y*S_y1 + b_z*S_z1;
            Sp_z =-n_x*S_x1 - n_y*S_y1 - n_z*S_z1;
            
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
            
            
            if (l == 0)
            {
                // Filter:
                int m = hData[i].Filter;
                // Difference between the observational and model magnitudes:
                double y = hData[i].V - Vmod;                    
                sum_y2[m] = sum_y2[m] + y*y*hData[i].w;
                sum_y[m] = sum_y[m] + y*hData[i].w;
                sum_w[m] = sum_w[m] + hData[i].w;
            }
            else
            {
                //                    if (m==0)
                //                    fprintf(fp, "%20.12lf %20.12lf %20.12lf %f %d\n", MJD1, Vmod+y_avr_min[0], hData[i].V+y_avr_min[0]-y_avr_min[m], sqrt(sgm2[i]), hData[i].Filter);
                fprintf(fpm, "%20.12lf %20.12lf %20.12lf\n", MJD1, Vmod+y_avr_min[0], (MJD1-hMJD0)/P);
                if (i < N_data)
                    fprintf(fpd, "%20.12lf %20.12lf %f %c %20.12lf\n", hData[i].MJD+hMJD0, hData[i].V+y_avr_min[0]-y_avr_min[hData[i].Filter], 1.0/sqrt(hData[i].w), all_filters[hData[i].Filter], hData[i].MJD/P);                    
            }
            
            
        } // data points loop
        
        if (l == 1)
            break;
        
        double chi2m;
        chi2a=0.0;    
        for (m=0; m<N_filters; m++)
        {
            // Chi^2 for the m-th filter:
            chi2m = sum_y2[m] - sum_y[m]*sum_y[m]/sum_w[m];
            chi2a = chi2a + chi2m;
        }   
        
        chi2_min = chi2a / (N_data - N_PARAMS - N_filters);
        for (m=0; m<N_filters; m++)
            y_avr_min[m] = sum_y[m]/sum_w[m];
        
        
        if (l == 0)
        {
            gettimeofday (&tdr1, NULL);
            timeval_subtract (&cputime, &tdr1, &tdr0);
            printf ("CPU time: %.2f ms\n", cputime*1000);
        }
    }  // l loop
    
    fclose (fpd);
    fclose (fpm);
    printf("Filter corrections: \n");
    for (m=0; m<N_filters; m++)
        printf("%c %f\n",all_filters[m], y_avr_min[m]);
    
    printf("chi2_min=%f\n", chi2_min);
    return 0;
}
