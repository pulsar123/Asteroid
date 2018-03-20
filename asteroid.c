/* Program to test the asteroid brightness calculation algorithm to be used with the ABC (Asteroid Brightness in CUDA) simulation package.
 *   The goal is to simulate the brigntess curve of the first interstellar asteroid  1I/2017 U1.   
 */

#include <sys/time.h>
#include <unistd.h>
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
    CHI_FLOAT chi2_tot=1e32;
    int i;
    int useGPU = 1;
    cudaStream_t  ID[2];
    
    // Observational data:
    int N_data; // Number of data points
    int N_filters; // Number of filters used in the data        
    
    if (argc == N_PARAMS0+2)
    {
        params.b = atof(argv[2]);
        params.P = atof(argv[3]);
        params.c = atof(argv[4]);
        params.theta = atof(argv[5]);
        params.cos_phi = atof(argv[6]);
        params.phi_a0 = atof(argv[7]);
#ifdef TUMBLE
        params.P_pr = atof(argv[8]);
        params.theta_pr = atof(argv[9]);
        params.phi_n0 = atof(argv[10]);
#endif        
#ifndef DEBUG        
        useGPU = 0;
#endif        
    }
    else if (argc != 3)
    {
        printf("Arguments: obs_file  results_file\n");
        printf("or\n");
        printf("Arguments: obs_file  list_of_parameters\n");
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
    read_data(argv[1], &N_data, &N_filters);
    
    
    #ifdef GPU    
    if (useGPU)
    {
//        fp = fopen(argv[2], "w");

        int N_threads = N_BLOCKS * BSIZE;
                
        gpu_prepare(N_data, N_filters, N_threads);
        


#ifdef SIMPLEX
        printf("\n*** Simplex optimization ***\n\n");
        printf("  N_threads = %d\n", N_threads);        
        
// &&&        
        CHI_FLOAT hLimits[2][N_PARAMS];
        int iparam = -1;
        int iPpr = -1;
        // Limits for each parameter during optimization:
        
        // b
        iparam++;
        hLimits[0][iparam] = 0.02;
        hLimits[1][iparam] = 0.9;
#ifdef LOG_BC
        hLimits[0][iparam] = log(hLimits[0][iparam]);
        hLimits[1][iparam] = log(hLimits[1][iparam]);
#endif        
        
        // frequency 1/P (1/days) 0...10
        iparam++;
        hLimits[0][iparam] = 24.0/8.5;
        hLimits[1][iparam] = 24.0/6;
        
        // Theta
        iparam++;
        hLimits[0][iparam] = 0.001/RAD;
        hLimits[1][iparam] = 179.999/RAD;
        
        // cos_phi
        iparam++;
        hLimits[0][iparam] = -0.999;
        hLimits[1][iparam] = 0.999;
        
        // phi_a0
        iparam++;
        hLimits[0][iparam] = 0.001;
        hLimits[1][iparam] = 2.0*PI-0.001;

#ifndef SYMMETRY        
        // c (not used in SYMMETRY modes)
        iparam++;
        hLimits[0][iparam] = hLimits[0][0];
        hLimits[1][iparam] = hLimits[1][0];
#endif        

#ifdef TUMBLE
        // frequency 1/P_pr (1/days) 0...10
        iparam++;
        iPpr = iparam;
        hLimits[0][iparam] = 24.0/240;
        hLimits[1][iparam] = 24.0/1.0;
        
        // Theta_pr
        iparam++;
        hLimits[0][iparam] = 0.001/RAD;
        hLimits[1][iparam] = 180.0/RAD;
        
        // phi_n0
        iparam++;
        hLimits[0][iparam] = 0.0;
        hLimits[1][iparam] = 2.0*PI;
#endif        
        
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

        ERR(cudaMemcpyToSymbol(dLimits, hLimits, 2*N_PARAMS*sizeof(CHI_FLOAT), 0, cudaMemcpyHostToDevice));                
        
   // Initializing the device random number generator:
        curandState* d_states;
        ERR(cudaMalloc ( &d_states, N_BLOCKS*BSIZE*sizeof( curandState ) ));
    // setup seeds, initialize d_f
        setup_kernel <<< N_BLOCKS, BSIZE >>> ( d_states, (unsigned long)(time(NULL)), d_f );
        //!!!
//        setup_kernel <<< N_BLOCKS, BSIZE >>> ( d_states, 1, d_f );

        ERR(cudaDeviceSynchronize());    

// Creating streams:
        for (i = 0; i < 2; ++i)
            ERR(cudaStreamCreate (&ID[i]));

#ifdef TIMING        
        cudaEvent_t start, stop;
        float elapsed;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
#endif        
        
#ifdef DEBUG
        debug_kernel<<<1, 1>>>(params, dData, N_data, N_filters);
#endif        
       
        // The kernel (using stream 0):
        chi2_gpu<<<N_BLOCKS, BSIZE, 0, ID[0]>>>(dData, N_data, N_filters, d_states, d_f, d_params, iPpr);

#ifdef TIMING
        cudaEventRecord(stop, 0);
        cudaEventSynchronize (stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        printf("GPU time: %.2f ms\n", elapsed);
        exit(0);
#endif        
        
        //!!!
        /*
        cudaDeviceSynchronize();
            cudaMemcpyAsync(h_f, d_f, N_threads * sizeof(CHI_FLOAT), cudaMemcpyDeviceToHost, ID[1]);
            cudaMemcpyAsync(h_params, d_params, N_threads * sizeof(struct parameters_struct), cudaMemcpyDeviceToHost, ID[1]);
        cudaDeviceSynchronize();
        fp = fopen(argv[2], "w");
                for (i=0; i<N_threads; i++)
                {
                    params = h_params[i];
                    fprintf(fp,"%13.6e ",  h_f[i]);
                    fprintf(fp,"%10.6f ",  params.b);
                    fprintf(fp,"%10.6f ",  params.P*24);
                    fprintf(fp,"%10.6f ",  params.c);
                    fprintf(fp,"%10.6f ",  params.cos_phi_b);
                    fprintf(fp,"%10.6f ",  params.theta);
                    fprintf(fp,"%10.6f ",  params.cos_phi);
                    fprintf(fp,"%10.6f ",  params.phi_a0);
#ifdef TUMBLE
                    fprintf(fp,"%10.6f ",  params.P_pr*24);
                    fprintf(fp,"%10.6f ",  params.theta_pr);
                    fprintf(fp,"%10.6f ",  params.phi_n0);
#endif                    
                    fprintf(fp,"\n");
                }
                fclose(fp);
exit(0);        
        */
        
        int not_done = 1;
        int count = 0;
        do
        {
            not_done = cudaStreamQuery(ID[0]);
            if (not_done)
                sleep(DT_DUMP);
            
            // Copying the results from GPU:
            /*
            ERR(cudaMemcpyAsync(h_f, d_f, N_threads * sizeof(CHI_FLOAT), cudaMemcpyDeviceToHost, ID[1]));
            ERR(cudaMemcpyAsync(h_params, d_params, N_threads * sizeof(struct parameters_struct), cudaMemcpyDeviceToHost, ID[1]));
            ERR(cudaStreamSynchronize(ID[1]));
*/
            ERR(cudaMemcpyFromSymbolAsync(&h_block_counter, d_block_counter, sizeof(int), 0, cudaMemcpyDeviceToHost, ID[1]));
            ERR(cudaStreamSynchronize(ID[1]));
            if (h_block_counter == 0)
                continue;
            ERR(cudaMemcpyAsync(h_f, d_f, h_block_counter * sizeof(CHI_FLOAT), cudaMemcpyDeviceToHost, ID[1]));
            ERR(cudaMemcpyAsync(h_params, d_params, h_block_counter * sizeof(struct parameters_struct), cudaMemcpyDeviceToHost, ID[1]));
            ERR(cudaMemcpyFromSymbolAsync(&h_min, d_min, sizeof(int), 0, cudaMemcpyDeviceToHost, ID[1]));
            ERR(cudaMemcpyFromSymbolAsync(&h_max, d_max, sizeof(int), 0, cudaMemcpyDeviceToHost, ID[1]));
            ERR(cudaMemcpyFromSymbolAsync(&h_sum, d_sum, sizeof(unsigned long long int), 0, cudaMemcpyDeviceToHost, ID[1]));
            ERR(cudaMemcpyFromSymbolAsync(&h_sum2, d_sum2, sizeof(unsigned long long int), 0, cudaMemcpyDeviceToHost, ID[1]));
            ERR(cudaStreamSynchronize(ID[1]));
            
            count++;
            if (count == N_WRITE || not_done==0)
            {
                count = 0;
                fp = fopen(argv[2], "w");
                for (i=0; i<h_block_counter; i++)
                {
                    params = h_params[i];
                    fprintf(fp,"%13.6e ",  h_f[i]);
                    fprintf(fp,"%10.6f ",  params.b);
                    fprintf(fp,"%10.6f ",  params.P*24);
                    fprintf(fp,"%10.6f ",  params.c);
                    fprintf(fp,"%10.6f ",  params.theta);
                    fprintf(fp,"%10.6f ",  params.cos_phi);
                    fprintf(fp,"%10.6f ",  params.phi_a0);
#ifdef TUMBLE
                    fprintf(fp,"%10.6f ",  params.P_pr*24);
                    fprintf(fp,"%10.6f ",  params.theta_pr);
                    fprintf(fp,"%10.6f ",  params.phi_n0);
#endif                    
                    fprintf(fp,"\n");
                }
                fclose(fp);
            }
            
            // Finding the best result between all threads:        
            int i_best = 0;
            chi2_tot = 1e32;
            int Nresults = 0;
            for (i=0; i<h_block_counter; i++)
            {
                if (h_f[i] < 1e29)
                    Nresults++;
                if (h_f[i] < chi2_tot)
                {
                    chi2_tot = h_f[i];
                    i_best = i;
                }
            }
            
            // Priting the best result:
            params = h_params[i_best];
            printf("%13.6e ",  h_f[i_best]);
            printf("%10.6f ",  params.b);
            printf("%10.6f ",  params.P*24);            
            printf("%10.6f ",  params.c);
            printf("%10.6f ",  params.theta);
            printf("%10.6f ",  params.cos_phi);
            printf("%10.6f ", params.phi_a0);
#ifdef TUMBLE
            printf("%10.6f ",  params.P_pr*24);
            printf("%10.6f ",  params.theta_pr);
            printf("%10.6f ",  params.phi_n0);
#endif             
            printf("%d ",  h_block_counter); // Number of finished blocks
            printf("%d ",  h_min);  // Min and max number of Simplex steps in all finished blocks
            printf("%d ",  h_max);
            double Nth = (double)(h_block_counter) * BSIZE;
            printf("%lf ",  (double)h_sum / Nth); // Average number of simplex steps
            printf("%lf ",  sqrt(((double)h_sum2 - 1.0/Nth * (double)h_sum * (double)h_sum)    / (Nth-1.0))); // std for the number of  simplex steps
            printf("\n");
            fflush(stdout);
        }
        while(not_done != 0);
        



/*
        for (i = 0; i < 2; ++i)
            ERR(cudaStreamDestroy (&ID[i]));
        ERR(cudaMemcpy(h_f, d_f, N_threads * sizeof(float), cudaMemcpyDeviceToHost);
        ERR(cudaMemcpy(h_params, d_params, N_threads * sizeof(struct parameters_struct), cudaMemcpyDeviceToHost);
        ERR(cudaDeviceSynchronize());
            
            for (i=0; i<N_threads; i++)
            {
                params = h_params[i];
                fprintf(fp,"%13.6e ",  h_f[i]);
                fprintf(fp,"%10.6f ",  params.b);
                fprintf(fp,"%10.6f ",  params.P*24);
                fprintf(fp,"%10.6f ",  params.c);
                fprintf(fp,"%10.6f ",  params.cos_phi_b);
                fprintf(fp,"%10.6f ",  params.theta);
                fprintf(fp,"%10.6f ",  params.cos_phi);
                fprintf(fp,"%10.6f\n", params.phi_a0);
            }
            fflush(fp);
*/






        
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
        fclose(fp);
#endif        
        
    }
    #endif    
    
    // CPU based chi^2:
    double chi2_cpu;
    chi2(N_data, N_filters, params, &chi2_cpu);    
    
    return 0;  
}
