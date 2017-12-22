/* Miscallaneous routines
 * 
*/
#include "asteroid.h"

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
