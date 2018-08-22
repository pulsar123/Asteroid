
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_combination.h>

// Matching the observed periodogram to the model (synthetic) ones

//  gcc -O2 freq.c -o freq -lgsl -lgslcblas -lm

int main (int argc,char **argv)
{
    char temp[10];
    int LAM, Nobs;
    double Ppsi, Pphi;
    
    // Maximum number of observed frequences:
    const int Nobs_max = 10;
    // Number of model frequencies:
    const int Nmod = 5;
    // Largest number of characters in the model file
    const int Nchar_max = 512;
    
    double f_obs[Nobs_max];
    double f_mod[Nmod];
    char buffer[Nchar_max];
    
    if (argc != 5)
    {
        printf ("Syntax: obs_data  model_data  output  Npeaks\n");
        exit (1);
    }

    int Npeaks = atoi(argv[4]);
    
    if (Npeaks < 2)
    {
        printf ("Npeaks should be >= 2!\n");
        exit(1);
    }
    
    FILE * fp_obs = fopen(argv[1], "r");
    FILE * fp_mod = fopen(argv[2], "r");
    FILE * fp_out = fopen(argv[3], "w");

    int i = 0;
    while (fgets(buffer, Nchar_max, fp_obs) != NULL)
    {
        sscanf(buffer, "%lf", &f_obs[i]);
        f_obs[i] = log10(f_obs[i]);
        i++;
    }
    Nobs = i;
    
    if (Npeaks > Nobs)
    {
        printf("Npeaks > Nobs!\n");
        exit(1);
    }

    gsl_combination * obs_comb = gsl_combination_alloc(Nobs, Npeaks);
    size_t * obsA = gsl_combination_data(obs_comb);
    
    // Going through the model file one line (one model) at a time:
    while (fgets(buffer, Nchar_max, fp_mod) != NULL)
    {
        // Reading one model line:
        sscanf(buffer, "%d %lf %lf %s %s %s %s %s %lf %lf %lf %lf %lf", &LAM, &Ppsi, &Pphi, temp, temp, temp, temp, temp, &f_mod[0], &f_mod[1], &f_mod[2], &f_mod[3], &f_mod[4] );

        int Nmod1 = 0;
        for (int i=0; i<Nmod; i++)
            if (f_mod[i] > 0)
                Nmod1++;
        if (Nmod1 < Npeaks)
            // Not enough of model peaks; skipping the model
            continue;
        
        // Sorting the model frequencies, getting rid of negative values
        for (int j=0; j<Nmod-1; j++)
        {
            double fmin = 1e30;
            int kmin = -1;
            for (int k=j; k<Nmod; k++)
            {
                if ((f_mod[k] < fmin) && f_mod[k]>0)
                {
                    fmin = f_mod[k];
                    kmin = k;
                }
            }
                        
            if (kmin >= 0)
                {
                    double ftemp = f_mod[j];
                    f_mod[j] = f_mod[kmin];
                    f_mod[kmin] = ftemp;
                }                        
        }
        
        for (int i=0; i<Nmod1; i++)
            f_mod[i] = log10(f_mod[i]);
            
        gsl_combination * mod_comb = gsl_combination_alloc(Nmod1, Npeaks);
        size_t * modA = gsl_combination_data(mod_comb);
        
        double dist1;
        double std_min = 1e30;
        
        // Outer loop goes through all combinations Npeaks out of Nobs, for observed data
        gsl_combination_init_first(obs_comb);
        do
        {
            // Inner while loop goes through all combinations Npeaks out of Nmode1, for the model
            gsl_combination_init_first(mod_comb);
            do
            {
                // Computing the mean distance between the model and the observations:
                double dist = 0.0;
                for (int k=0; k<Npeaks; k++)
                {
                    dist = dist + f_mod[modA[k]]-f_obs[obsA[k]];
                }
                dist = dist / Npeaks;
                
                // Computing the std between the model and observations relative to the mean:
                double s2 = 0.0;
                for (int k=0; k<Npeaks; k++)
                {
                    double d = f_mod[modA[k]]-f_obs[obsA[k]] - dist;
                    s2 = s2 + d * d;
                }
                double std = sqrt(s2/(Npeaks-1));
                if (std < std_min)
                {
                    std_min = std;
                    dist1 = dist;
                }
            }
            while (gsl_combination_next (mod_comb) == GSL_SUCCESS);            
        }
        while (gsl_combination_next (obs_comb) == GSL_SUCCESS);
        
        gsl_combination_free(mod_comb);        
        
        fprintf(fp_out, "%f %f %s", std_min, dist1, buffer);
        
    } // end of while loop to read lines from the model file
    
    
    
    gsl_combination_free(obs_comb);
    fclose(fp_obs);
    fclose(fp_mod);
    fclose(fp_out);        
    
    return 0;
    
}
