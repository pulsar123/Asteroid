
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Frequency analysis of periodograms produced by asteroid.c

int main (int argc,char **argv)
{
    
    if (argc == 1)
    {
        printf ("Syntax: Ppsi Pphi f1 f2 ... f5\n");
        printf ("Units: hrs (Ppsi, Pphi); 1/d (f#)\n");
        exit (1);
    }
    
    double f_psi = 24.0/atof(argv[1]);
    double f_phi = 24.0/atof(argv[2]);
    
    double f[5];
    
    for (int i=0; i<5; i++)
    {
        f[i] = atof(argv[3+i]);
    }
    
    // Largest frequency harmonic considered:
    const int Nmax = 6;
    // Maximum allowed fractional deviation of frequencies:
    const double DF_MAX = 0.01;
    
    int kbest, lbest;
    double df_min;
    // Cycle for the observed frequency peak in the periodogram (from most to least significant):
    for (int i=0; i<5; i++)
    {
        kbest = 100;
        lbest = 100;
        df_min = 1e30;
        // Skipping non-existing peaks:
        if (f[i] > 0.0)
        {
            // Double cycles for different harmonics of f_psi and f_phi:
            for (int k=0; k<=Nmax; k++)
            {
                for (int l=0; l<=Nmax; l++)
                {
                    if (k<0 && l<0)
                        continue;
                    if (k==0 && l<=0 || l==0 && k<=0)
                        continue;
                    double ff = fabs(k*f_psi+l*f_phi);
                    double df = fabs((f[i]-ff)/f[i]);
                    if (df < df_min)
                    {
                        df_min = df;
                        kbest = k;
                        lbest = l;
                    }
                }
            }            
        }
        if (df_min < DF_MAX)
            // We found a good match
            printf ("%+2d,%+2d ", kbest, lbest);
        else
            printf ("wrong ");                        
    }
    
    printf ("\n");
    
    return 0;
    
}
