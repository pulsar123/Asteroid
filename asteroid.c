/* Program to test the asteroid brightness calculation algorithm to be used with the ABC (Asteroid Brightness in CUDA) simulation package.
 *   The goal is to simulate the brigntess curve of the first interstellar asteroid  1I/2017 U1.   
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define MAIN
#include "asteroid.h"

int main (int argc,char **argv)
{
    double chi2_tot=1e32;
    long int iloc_tot;
    int i;
    
    // Observational data:
    int N_data; // Number of data points
    int N_filters; // Number of filters used in the data        
    
    Is_GPU_present();
    
    // Reading input paameters files
    //    read_input_params();
    
    // Reading all input data files, allocating and initializing observational data arrays   
    read_data("obs.dat", &N_data, &N_filters);
    
    #ifdef GPU    
    long int Nloc = N_THETA * N_COS_PHI * N_PHI_A;
    int Nglob = N_B * N_P;
    
    gpu_prepare(N_data, N_filters);
    
    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    
    // We use Y and Z grid dimensions to store iglob:
    //    int Z_dim = Nglob / deviceProp.maxGridSize[1];
    //    int Y_dim = Nglob % deviceProp.maxGridSize[1];
    
    // We use a combination of X grid dimension, BSIZE threads and N_SERIAL loops to store iloc:
    // Make sure Nloc/N_SERIAL is integer dividable!
    //    int X_dim = (Nloc/N_SERIAL+BSIZE-1) / BSIZE;
    //    dim3 Dims(X_dim, Y_dim, Z_dim);
    
    int N_threads = N_BLOCKS * BSIZE;
    int N_serial = (Nloc+N_threads-1) / N_threads;
    printf ("N_threads=%d; N_serial=%d\n", N_threads, N_serial);
    
    int iglob = 0;
    int iglob_tot = 0;
    int i_b, i_P, i_b_min, i_P_min;
    int init = 1;
    
    for (i_b=0; i_b<N_B; i_b++)
    {
        for (i_P=0; i_P<N_P; i_P++)
        {
            iglob = i_b*N_P + i_P;
            
            if (init == 1)
                cudaEventRecord(start, 0);
            
            // The kernel:
            chi2_gpu<<<N_BLOCKS, BSIZE>>>(dData, N_data, N_filters, Nloc, iglob, N_serial, d_chi2_min, d_iloc_min);
            // Copying the results (one per block) to CPU:
            cudaMemcpy(h_chi2_min, d_chi2_min, N_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_iloc_min, d_iloc_min, N_BLOCKS * sizeof(long int), cudaMemcpyDeviceToHost);
            
            double chi2_loc = 1e31;
            long int iloc = 0;

// Finding the best result between all the blocks:            
            for (i=0; i<N_BLOCKS; i++)
            {
                if (h_chi2_min[i] < chi2_loc)
                {
                    chi2_loc =  h_chi2_min[i];
                    iloc = h_iloc_min[i];
                }
            }

            // Globally the best result:
            if (chi2_loc < chi2_tot)
            {
                chi2_tot =  chi2_loc;
                iloc_tot = iloc;
                iglob_tot = iglob;
            }
            
            if (init == 1)
            {
                cudaEventRecord(stop, 0);
                cudaEventSynchronize (stop);
                cudaEventElapsedTime(&elapsed, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                printf("GPU time: %.2f ms\n", elapsed);
            }
            
            init = 0;
        } // i_P
    }  // i_b
    
    cudaDeviceSynchronize();
    struct parameters_struct params;
    iloc_to_params(&iloc_tot, &params);
    iglob_to_params(&iglob_tot, &params);
    printf("GPU chi2_min=%lf, iglob=%d, iloc=%lu\n", chi2_tot, iglob_tot, iloc_tot);
    printf("b=%lf\n", params.b);
    printf("P=%lf\n", params.P);
    printf("theta=%lf\n", params.theta);
    printf("cos_phi=%lf\n", params.cos_phi);
    printf("phi_a0=%lf\n", params.phi_a0);
    
    #endif    
    
    // CPU based chi^2:
    chi2(N_data, N_filters, &chi2_tot);    
    
    return 0;  
}
