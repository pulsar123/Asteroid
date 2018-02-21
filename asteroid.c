/* Program to test the asteroid brightness calculation algorithm to be used with the ABC (Asteroid Brightness in CUDA) simulation package.
 *   The goal is to simulate the brigntess curve of the first interstellar asteroid  1I/2017 U1.   
 */

#include <sys/time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#define MAIN
#include "asteroid.h"

int main (int argc,char **argv)
{
    FILE *fp;
    struct parameters_struct params;
    float chi2_tot=1e32;
    int i;
    int useGPU = 1;
    
    // Observational data:
    int N_data; // Number of data points
    int N_filters; // Number of filters used in the data        
    
    if (argc == 8)
    {
        params.b = atof(argv[1]);
        params.P = atof(argv[2]);
        params.c = atof(argv[3]);
        params.cos_phi_b = atof(argv[4]);
        params.theta = atof(argv[5]);
        params.cos_phi = atof(argv[6]);
        params.phi_a0 = atof(argv[7]);
        useGPU = 0;
    }
    else if (argc != 1)
    {
        printf("Wrong arguments!\n");
        exit(1);
    }
    
    
    #ifdef GPU
    if (useGPU)
    {
        Is_GPU_present();
    }
    #endif    
    
    // Reading input paameters files
    //    read_input_params();
    
    // Reading all input data files, allocating and initializing observational data arrays   
    read_data("obs.dat", &N_data, &N_filters);
    
    
    #ifdef GPU    
    if (useGPU)
    {
        fp = fopen("results.dat", "w");

        int N_threads = N_BLOCKS * BSIZE;
                
        gpu_prepare(N_data, N_filters, N_threads);
        


#ifdef SIMPLEX
        
        float hLimits[2][N_PARAMS];
        int iparam;
        // Limits for each parameter during optimization:
        // b
        iparam = 0;
        hLimits[0][iparam] = 0.08;
        hLimits[1][iparam] = 0.28;
        
        // P
        iparam = 1;
        hLimits[0][iparam] = 6.5/24;
        hLimits[1][iparam] = 8.5/24;
        
        // Theta
        iparam = 2;
        hLimits[0][iparam] = 0.001/RAD;
        hLimits[1][iparam] = 180.0/RAD;
        
        // cos_phi
        iparam = 3;
        hLimits[0][iparam] = -1.0;
        hLimits[1][iparam] = 0.999;
        
        // phi_a
        iparam = 4;
        hLimits[0][iparam] = 0.0;
        hLimits[1][iparam] = 2.0*PI;
        
// Normalizing parameters to delta=1 range: ???
        /*
        float delta;
        for (i=0; i<N_PARAMS; i++)
        {
            delta = hLimits[1,i] - hLimits[0,i];
            hLimits[2,i] = delta;
            hLimits[0,i] = hLimits[0,i] / delta;
            hLimits[1,i] = hLimits[1,i] / delta;
        }
        */

        ERR(cudaMemcpyToSymbol(dLimits, hLimits, 2*N_PARAMS*sizeof(float), 0, cudaMemcpyHostToDevice));                
        
   // Initializing the device random number generator:
        curandState* d_states;
        cudaMalloc ( &d_states, N_threads*sizeof( curandState ) );
    // setup seeds, initialize d_f
//        setup_kernel <<< N_BLOCKS, BSIZE >>> ( d_states, time(NULL), d_f );
        //!!!
        setup_kernel <<< N_BLOCKS, BSIZE >>> ( d_states, 0, d_f );

        // The kernel:
        chi2_gpu<<<N_BLOCKS, BSIZE>>>(dData, N_data, N_filters, d_states, d_f, d_params);

// Copying the results from GPU:
        ERR(cudaMemcpy(h_f, d_f, N_threads * sizeof(float), cudaMemcpyDeviceToHost));
        ERR(cudaMemcpy(h_params, d_params, N_threads * sizeof(struct parameters_struct), cudaMemcpyDeviceToHost));

// Finding the best result between all threads:        
        int i_best = 0;
        for (i=0; i<N_threads; i++)
        {
            if (h_f[i] < chi2_tot)
            {
                chi2_tot = h_f[i];
                i_best = i;
            }
        }
        
// Writing the best result to file:
        params = h_params[i_best];
        fprintf(fp,"b=%lf\n", params.b);
        fprintf(fp,"P=%lf\n", params.P*24);
        fprintf(fp,"theta=%lf\n", params.theta);
        fprintf(fp,"cos_phi=%lf\n", params.cos_phi);
        fprintf(fp,"phi_a0=%lf\n", params.phi_a0);
        
#else        
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
        
        long int Nloc = N_THETA * N_COS_PHI * N_PHI_A;
        long int iloc_tot;
        int N_serial = (Nloc+N_threads-1) / N_threads;
        printf ("N_threads=%d; N_serial=%d\n", N_threads, N_serial);
        
        int iglob = 0;
        int iglob_tot = 0;
        int i_b, i_P;
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
                ERR(cudaMemcpy(h_chi2_min, d_chi2_min, N_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));
                ERR(cudaMemcpy(h_iloc_min, d_iloc_min, N_BLOCKS * sizeof(long int), cudaMemcpyDeviceToHost));
                ERR(cudaDeviceSynchronize());
                
                float chi2_loc = 1e31;
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
                printf("iglob=%d, chi2=%lf\n",iglob,chi2_loc);
                
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
                
                struct parameters_struct params;
                iloc_to_params(&iloc, &params);
                iglob_to_params(&iglob, &params);
                fprintf(fp,"%e %7d %9lu %10.5lf %10.5lf %10.5lf %10.5lf %10.5lf %10.5lf %10.5lf\n",
                        chi2_loc, iglob, iloc, params.b, params.P*24, params.c, params.cos_phi_b, params.theta, params.cos_phi, params.phi_a0);
                
                init = 0;
            } // i_P
        }  // i_b
        
        cudaDeviceSynchronize();    
        iloc_to_params(&iloc_tot, &params);
        iglob_to_params(&iglob_tot, &params);
        
        printf("GPU chi2_min=%e, iglob=%d, iloc=%lu\n", chi2_tot, iglob_tot, iloc_tot);
        printf("b=%lf\n", params.b);
        printf("P=%lf\n", params.P*24);
        printf("theta=%lf\n", params.theta);
        printf("cos_phi=%lf\n", params.cos_phi);
        printf("phi_a0=%lf\n", params.phi_a0);
#endif        
        
        fclose(fp);
    }
    #endif    
    
    // CPU based chi^2:
    double chi2_cpu;
    chi2(N_data, N_filters, params, &chi2_cpu);    
    
    return 0;  
}
