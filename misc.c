/* Miscallaneous routines
 * 
 */
#include "asteroid.h"

// Used with qsort:
int cmpdouble (const void * a, const void * b) {
    if (*(double*)a > *(double*)b)
        return 1;
    else if (*(double*)a == *(double*)b)
        return 0;
    else
        return -1;
}
//int cmpint (const void * a, const void * b) {
//   return ( *(int*)a - *(int*)b );
//}


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
     *     tv_usec is certainly positive. */
    result0.tv_sec = x->tv_sec - y->tv_sec;
    result0.tv_usec = x->tv_usec - y->tv_usec;
    *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;
    
    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}


int minima(struct obs_data * dPlot, double * Vm, int Nplot)
// Finding minima and computing periodogramm
{
    
    // Maximum number of minima:    
    #define NMIN_MAX 10000
    double t[NMIN_MAX];
    
    // Finding all minima:    
    int N = 0;    
    for (int i=1; i<=Nplot-2; i++) 
    {
        if (Vm[i]>Vm[i-1] && Vm[i]>=Vm[i+1]) 
            // We just found a brightness minimum (V maximum)
        {
            N++;
            if (N > NMIN_MAX)
            {
                printf ("N>NMIN_MAX!\n");
                exit(1);
            }
#ifdef PARABOLIC_MAX
            //Fitting a parabola to the three points (i-1), i, (i+1)
            double a = ((Vm[i+1]-Vm[i])/(dPlot[i+1].MJD-dPlot[i].MJD) - (Vm[i]-Vm[i-1])/(dPlot[i].MJD-dPlot[i-1].MJD)) / (dPlot[i+1].MJD-dPlot[i-1].MJD);
            double b = (Vm[i]-Vm[i-1])/(dPlot[i].MJD-dPlot[i-1].MJD) - a*(dPlot[i].MJD+dPlot[i-1].MJD);
            // Maximum point for the parabola:
            t[N-1] = -b/2.0/a;
#else
            t[N-1] = dPlot[i].MJD;
#endif            
        }
    }
    
    // Now N is the number of minima found; the minima values (days) are in t[0...N-1].
    
    // Total number of time intervals in t[]:
    int M = N*(N-1)/2;
    
    double * dt = (double*)malloc(M*sizeof(double));
    int * i1 = (int*)malloc(M*sizeof(int));
    int * i2 = (int*)malloc(M*sizeof(int));
    
    // Finding all time intervals:
    int k = -1;
    for (int i=0; i<=N-2; i++) 
    {
        for (int j=i+1; j<=N-1; j++) 
        {
            k++;
            if (k >= M)
            {
                printf ("k >= M!\n");
                exit(1);
            }
            dt[k]=t[j]-t[i];
            i1[k] = i;
            i2[k] = j;
        }
    }
    // Actual number of time intervals:
    M = k + 1;
    
    // Sorting dt:
    qsort(dt, M, sizeof(double), cmpdouble);
    
    int NN = 10000;   // Number of histogram bins
    double sgm = 0.1;  // Maximum fractional deviation of a given frequency from a histogram fr value to be counted as a match
    double dt0 = 1.0/24.0;  // Minimum histogram dt
    double dt2 = 80.0/24.0; // Maximum histogram dt
    double s0 = 3;  // How many sigmas above the histogram noise to count as a significant peak
    
    double fr0 = 1/dt0;
    double fr2 = 1/dt2;
    double * fr = (double*)malloc(NN*sizeof(double));
    double * H = (double*)malloc(NN*sizeof(double));
    int * marked = (int*)malloc(NN*sizeof(int));
        
    double m = 0.0;
    for (int i=0; i<NN; i++) 
    {
        fr[i] = fr2 + (double)i/(double)(NN-1) * (fr0-fr2);
        H[i] = 0.0;
        double dt1 = 1/fr[i];
        for (int j=0; j<M; j++)
        {
            double dv=fabs(dt[j]/dt1-int(dt[j]/dt1+0.5));
#ifdef MINIMA_SPLINE
            // Using M4 B-spline function instead of step function
            double q = dv / sgm;
            double H1;
            double q1 = 1.0 - q;
            double q2 = 2.0 - q;
            if (q <= 1.0)
                H1 = 0.25*q2*q2*q2 - q1*q1*q1;
            else if (q <= 2.0)
                H1 = 0.25*q2*q2*q2;
            else
                H1 = 0.0;
            H[i] = H[i] + H1;
#else            
            if (dv < sgm)
                // Current time interval is within fractional sgm from the histogram bin value, 1/fr[i], so we are counting it as good
            {
                H[i] = H[i] + 1.0;
            }        
#endif            
        }
        m = m + H[i];
    }
    
    // The histogram has been computed - H(fr)
    
    // Histogram mean:
    m = m / NN;    
    double s = 0;
    for (int i=0; i<NN; i++) 
    {
        double dev = H[i] - m;
        s = s + dev*dev;
    }
    if (NN < 2)
    {
        printf ("NN < 2!\n");
        exit(1);
    }
    // Histogram std:
    s = sqrt(s/(NN-1));
    
#ifdef MINIMA_PRINT
    FILE * fp_minima = fopen("min_profile.dat","w");
#endif    
    // Normalizing the histogram by mean and std:
    for (int i=0; i<NN; i++) 
    {
        H[i] = (H[i]-m)/s;
        marked[i] = 0;
        #ifdef MINIMA_PRINT
        fprintf(fp_minima, "%lf %lf\n", fr[i], H[i]);
        #endif
    }
#ifdef MINIMA_PRINT
    fclose(fp_minima);
#endif    
    // Cluster boundaries are determined by this H lowest value:
    double cl_min = s0;
    
    // Finding NCL_MAX top frequencies clusters:
    int Ncl = 0;
    
    for (int j=0; j<NCL_MAX; j++)
    {
        double max = -100;
        int imax = -1;
        cl_fr[j] = -1.0;
        
        // Serching for the current unmarked maximum value:
        for (int i=0; i<NN; i++) 
        {
            if (marked[i]==0 && H[i]>max) {
                max = H[i];
                imax = i;
            }
        }
        if (imax > -1 && H[imax]>=cl_min) 
            // If we found a maximum, and this maximum is above the threshold cl_min - we accept it
        {
            // imax marks the top of the newly found cloud:
            cl_fr[Ncl] = fr[imax];
            cl_H[Ncl] = H[imax];
            Ncl++;
            
            // Finding and marking the right cloud extent:
            int bad = 0;
            int i = imax;
            while (bad == 0) 
            {
                if (H[i] > cl_min)
                    marked[i] = 1;
                else
                    bad = 1;
                i++;
                if (i == NN)
                    bad = 1;
            }
            
            // Finding and marking the left cloud extent:
            bad = 0;
            i = imax-1;
            while (bad == 0)
            {
                if (H[i] > cl_min)
                    marked[i] = 1;
                else
                    bad = 1;
                i--;
                if (i == -1)
                    bad = 1;
            }
            
        }
        
    }            
    
    free(dt);
    free(i1);
    free(i2);
    free(fr);
    free(H);
    free(marked);
    
    return 0;
}
