/* Miscallaneous routines
 * 
*/
#include "asteroid.h"

__device__ __host__ void iglob_to_params(int * iglob, struct parameters_struct * params)
// Converting from the global CUDA index iglob to actual parameters, for both GPU and CPU code
{
    int i, rest;
    
    i = *iglob % N_P;
    params->P = (double)i/(double)N_P * (P2 - P1) + P1;
    rest = *iglob / N_P;

    params->b = (double)rest/(double)N_B * (B2 - B1) + B1;
    params->c = params->b;
    
    return;   
}

__device__ __host__ void iloc_to_params(long int * iloc, struct parameters_struct * params)
// Converting from the local CUDA index iloc to actual parameters, for both GPU and CPU code
{
    long int i, rest;
    
    i = *iloc % N_PHI_A;
    params->phi_a0 = (double)i/(double)N_PHI_A * (PHI_A2 - PHI_A1) + PHI_A1;
    rest = *iloc / N_PHI_A;

    i = rest % N_COS_PHI;
    params->cos_phi = (double)i/(double)N_COS_PHI * (COS_PHI2 - COS_PHI1) + COS_PHI1;
    rest = rest / N_COS_PHI;
        
    params->theta = (double)rest/(double)N_THETA * (THETA2 - THETA1) + THETA1;

    params->cos_phi_b = 0.0;

    return;   
}


int quadratic_interpolation(double MJD, double *E_x1,double *E_y1,double *E_z1, double *S_x1,double *S_y1,double *S_z1)
{
    double rr[3];
                
                rr[0] = (MJD-MJD0[1]) * (MJD-MJD0[2]) / (MJD0[0]-MJD0[1]) / (MJD0[0]-MJD0[2]);
                rr[1] = (MJD-MJD0[0]) * (MJD-MJD0[2]) / (MJD0[1]-MJD0[0]) / (MJD0[1]-MJD0[2]);
                rr[2] = (MJD-MJD0[0]) * (MJD-MJD0[1]) / (MJD0[2]-MJD0[0]) / (MJD0[2]-MJD0[1]);
                *E_x1 = E_x0[0]*rr[0] + E_x0[1]*rr[1] + E_x0[2]*rr[2];
                *E_y1 = E_y0[0]*rr[0] + E_y0[1]*rr[1] + E_y0[2]*rr[2];
                *E_z1 = E_z0[0]*rr[0] + E_z0[1]*rr[1] + E_z0[2]*rr[2];
                *S_x1 = S_x0[0]*rr[0] + S_x0[1]*rr[1] + S_x0[2]*rr[2];
                *S_y1 = S_y0[0]*rr[0] + S_y0[1]*rr[1] + S_y0[2]*rr[2];
                *S_z1 = S_z0[0]*rr[0] + S_z0[1]*rr[1] + S_z0[2]*rr[2];
            
 return 0;   
}


int timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


