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
    #ifdef P_PHI
    double Pphi1, Pphi2;
    #endif
    #if defined(P_PSI) || defined(P_BOTH)
    double Ppsi1, Ppsi2;
    #endif
    #ifdef P_BOTH
    double P_phi;
    float hPphi;
    #endif   
    
    // Observational data:
    int N_data; // Number of data points
    int N_filters; // Number of filters used in the data        
    int Nplot = 0;
    
    
    if (argc == 10 || argc == 12)
    {
        params.theta_M = atof(argv[2]);
        params.phi_M = atof(argv[3]);
        params.phi_0 = atof(argv[4]);
        params.L = 48.0*PI/atof(argv[5]);        
        params.c_tumb = atof(argv[6]);
        params.b_tumb = atof(argv[7]);
        params.Es = atof(argv[8]);
        params.psi_0 = atof(argv[9]);
        if (argc == 12)
        {
            params.c = atof(argv[10]);
            params.b = atof(argv[11]);
        }
        Nplot = NPLOT; // Number of plot points
    }
    #ifdef REOPT
    else if (argc == 11 || argc == 13)
    {
        params.theta_M = atof(argv[3]);
        params.phi_M = atof(argv[4]);
        params.phi_0 = atof(argv[5]);
        params.L = 48.0*PI/atof(argv[6]);        
        params.c_tumb = atof(argv[7]);
        params.b_tumb = atof(argv[8]);
        params.Es = atof(argv[9]);
        params.psi_0 = atof(argv[10]);
        if (argc == 13)
        {
            params.c = atof(argv[11]);
            params.b = atof(argv[12]);
        }
        
    }
    #endif
    #ifdef P_PHI
    else if (argc == 5)
    {
        Pphi1 = atof(argv[3]);
        Pphi2 = atof(argv[4]);
    }
    #endif        
    #ifdef P_PSI
    else if (argc == 5)
    {
        Ppsi1 = atof(argv[3]);
        Ppsi2 = atof(argv[4]);
    }
    #endif        
    #ifdef P_BOTH
    else if (argc == 6)
    {
        Ppsi1 = atof(argv[3]);
        Ppsi2 = atof(argv[4]);
        P_phi = atof(argv[5]);
        hPphi = P_phi / 24.0 / (2*PI);
    }
    #endif        
    else if (argc != 3)
    {
        printf("Arguments: obs_file  results_file\n");
        printf("or\n");      
        printf("Arguments: obs_file  list_of_parameters\n");
        #ifdef P_PHI
        printf("or\n");      
        printf("Arguments: obs_file  results_file  Pphi1  Pphi2 (hrs)\n");
        #endif        
        #ifdef P_PSI
        printf("or\n");      
        printf("Arguments: obs_file  results_file  Ppsi1  Ppsi2 (hrs)\n");
        #endif        
        #ifdef P_BOTH
        printf("or\n");      
        printf("Arguments: obs_file  results_file  Ppsi1  Ppsi2  Pphi (hrs)\n");
        #endif        
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
    read_data(argv[1], &N_data, &N_filters, Nplot);
    
    
    #ifdef GPU    
    if (useGPU)
    {
        //        fp = fopen(argv[2], "w");
        
        int N_threads = N_BLOCKS * BSIZE;
        
        gpu_prepare(N_data, N_filters, N_threads, Nplot);
        
        // &&&        
        CHI_FLOAT hLimits[2][N_INDEPEND];
        int iparam = -1;
        // Limits for each independent model parameter during optimization:
        
        // (0) Theta_M (angle between barycentric Z axis and angular momentum vector M); range 0...pi
        iparam++;
        hLimits[0][iparam] = 0.001/RAD;
        hLimits[1][iparam] = 179.999/RAD;
        
        // (1) phi_M (polar angle for the angular momentum M in the barycentric FoR); range 0 ... 2*pi initially, can have any value during optimization
        iparam++;
        hLimits[0][iparam] = 0;
        hLimits[1][iparam] = 360.0/RAD;
        
        // (2) phi_0 (initial Euler angle for precession), 0...360 dgr initially, can have any value during optimization
        iparam++;
        hLimits[0][iparam] = 0/RAD;
        hLimits[1][iparam] = 360.0/RAD;
        
        // (3) Angular momentum L value, radians/day; if P is perdiod in hours, L=48*pi/P
        iparam++;
        hLimits[0][iparam] = 48.0*PI / 10; // 8.5
        hLimits[1][iparam] = 48.0*PI / 0.1; // 0.4    
        #ifndef REOPT
        // In P_PHI mode has a different meaning: 48*pi/Pphi2 ... 48*pi/Pphi1 (used to generate L)
        #ifdef P_PHI
        hLimits[0][iparam] = 48.0*PI / Pphi2;
        hLimits[1][iparam] = 48.0*PI / Pphi1;
        #endif
        // In P_PSI mode has a different meaning: Ppsi1 ... Ppsi2, days (used to derive L)
        #if defined(P_PSI) || defined(P_BOTH)
        hLimits[0][iparam] = Ppsi1/24.0;
        hLimits[1][iparam] = Ppsi2/24.0;
        #endif
        #endif    
        
        // (4) c_tumb (physical (tumbling) value of the axis c size; always smallest)
        iparam++;
        hLimits[0][iparam] = log(0.002);
        hLimits[1][iparam] = log(1.0);                
        
        ERR(cudaMemcpyToSymbol(dLimits, hLimits, 2*N_INDEPEND*sizeof(CHI_FLOAT), 0, cudaMemcpyHostToDevice));                
        #ifdef P_BOTH
        ERR(cudaMemcpyToSymbol(dPphi, &hPphi, sizeof(float), 0, cudaMemcpyHostToDevice));                
        #endif    
        
        if (Nplot == 0)
        {
            printf("\n*** Simplex optimization ***\n\n");
            printf("  N_threads = %d\n", N_threads);        
            
            // Initializing the device random number generator:
            curandState* d_states;
            ERR(cudaMalloc ( &d_states, N_BLOCKS*BSIZE*sizeof( curandState ) ));
            // setup seeds, initialize d_f
            setup_kernel <<< N_BLOCKS, BSIZE >>> ( d_states, (unsigned long)(time(NULL)), d_f);
            
            #ifdef REOPT
            ERR(cudaMemcpyToSymbol(d_params0, &params, sizeof(struct parameters_struct), 0, cudaMemcpyHostToDevice));
            #endif        
            
            ERR(cudaDeviceSynchronize());    
            
            // Creating streams:
//            for (i = 0; i < 2; ++i)
//                ERR(cudaStreamCreate (&ID[i]));
            
            int loop_counter = 0;
            
            // Infinite loop
            while (1)
            {                
                loop_counter++;
                
                #ifdef TIMING        
                cudaEvent_t start, stop;
                float elapsed;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                #endif        
                
                #ifdef DEBUG2
                debug_kernel<<<1, 1>>>(params, dData, N_data, N_filters);
                #endif        
                
                // The kernel (using stream 0):
                chi2_gpu<<<N_BLOCKS, BSIZE>>>(dData, N_data, N_filters, d_states, d_f, d_params);
                
                #ifdef TIMING
                cudaEventRecord(stop, 0);
                cudaEventSynchronize (stop);
                cudaEventElapsedTime(&elapsed, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                printf("GPU time: %.2f ms\n", elapsed);
                exit(0);
                #endif        
                
                if (loop_counter > 1)
                {
//                    ERR(cudaStreamSynchronize(ID[1]));
                    
                    fp = fopen(argv[2], "w");
                    for (i=0; i<N_BLOCKS; i++)
                    {
                        params = h_params[i];
                        fprintf(fp,"%13.6e ",  h_f[i]);
                        fprintf(fp,"%15.11f ",  params.theta_M);
                        fprintf(fp,"%15.11f ",  params.phi_M);
                        fprintf(fp,"%15.11f ",  params.phi_0);
                        fprintf(fp,"%15.11f ",  48*PI/params.L);
                        fprintf(fp,"%15.11f ",  params.c_tumb);
                        fprintf(fp,"%15.11f ",  params.b_tumb);
                        fprintf(fp,"%15.11f ",  params.Es);
                        fprintf(fp,"%15.11f ",  params.psi_0);
                        #ifdef BC
                        fprintf(fp,"%15.11f ",  params.c);
                        fprintf(fp,"%15.11f ",  params.b);
                        #endif                    
                        fprintf(fp,"\n");
                    }
                    fclose(fp);
                    
                    
                    // Finding the best result between all threads:        
                    int i_best = 0;
                    chi2_tot = 1e32;
                    int Nresults = 0;
                    for (i=0; i<N_BLOCKS; i++)
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
                    printf("%15.11f ",  params.theta_M);
                    printf("%15.11f ",  params.phi_M);
                    printf("%15.11f ",  params.phi_0);
                    printf("%15.11f ",  48*PI/params.L);
                    printf("%15.11f ",  params.c_tumb);
                    printf("%15.11f ",  params.b_tumb);
                    printf("%15.11f ",  params.Es);
                    printf("%15.11f ",  params.psi_0);  
                    #ifdef BC
                    printf("%15.11f ",  params.c);
                    printf("%15.11f ",  params.b);
                    #endif            
                    printf("\n");
                    fflush(stdout);
                }
                
                ERR(cudaDeviceSynchronize());
                
                ERR(cudaMemcpy(h_f, d_f, N_BLOCKS * sizeof(CHI_FLOAT), cudaMemcpyDeviceToHost));
                ERR(cudaMemcpy(h_params, d_params, N_BLOCKS * sizeof(struct parameters_struct), cudaMemcpyDeviceToHost));
                ERR(cudaDeviceSynchronize());
                
            }  // End of the while loop
            
        }  // End of Nplot=0 (simulation) module
        
        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        else
        {
            printf("\n*** Plotting ***\n\n");
            
            int NX;
            int NX1 = N_data/N_PARAMS+1;
            if (C_POINTS > NX1)
                NX = C_POINTS;
            else
                NX = NX1;
            
            #ifdef PROFILES        
            dim3 NB(NX, N_PARAMS);
            #else        
            //??? Not the proper way
            dim3 NB (1, 1);
            #endif        
            
            // Running the CUDA kernel to produce the plot data from params:
            chi2_plot<<<NB, BSIZE>>>(dData, N_data, N_filters, d_params, dPlot, Nplot, params, d_dlsq2);
            ERR(cudaDeviceSynchronize());
            
            ERR(cudaMemcpyFromSymbol(&h_Vmod, d_Vmod, Nplot*sizeof(double), 0, cudaMemcpyDeviceToHost));
            #ifdef PROFILES        
            ERR(cudaMemcpyFromSymbol(&h_chi2_plot, d_chi2_plot, sizeof(CHI_FLOAT), 0, cudaMemcpyDeviceToHost));
            ERR(cudaMemcpyFromSymbol(&h_chi2_lines, d_chi2_lines, sizeof(h_chi2_lines), 0, cudaMemcpyDeviceToHost));
            #endif        
            #ifdef LSQ
            ERR(cudaMemcpy(h_dlsq2, d_dlsq2, N_data*sizeof(double), cudaMemcpyDeviceToHost));
            #endif        
            ERR(cudaDeviceSynchronize());
            
            // Finding minima and computing periodogramm
            minima(dPlot, h_Vmod, Nplot);
            
            for (int j=0; j<NCL_MAX; j++)
                //            if (cl_fr[j] > 0.0)
                //        printf("%d %f %f %f %f\n", j, cl_fr[j], 24.0/cl_fr[j], 48.0/cl_fr[j], cl_H[j]);
                printf ("%f ", cl_fr[j]);
            for (int j=0; j<NCL_MAX; j++)
                printf ("%f ", cl_H[j]);
            printf("\n");
            
            
            //        double d2 = 0.0;
            /*
             *        for (i=0; i<N_data; i++)
             *            d2 = d2 + h_dlsq2[i];
             */
            double d2[6], w[6];
            int ind;
            for (i=0; i<6; i++)
            {
                d2[i] = 0.0;
                w[i] = 0.0;
            }
            for (i=0; i<N_data; i++)
            {
                if (hData[i].MJD < 0.5)
                    ind = 0;
                else if (hData[i].MJD < 1.5)
                    ind = 1;
                else if (hData[i].MJD < 2.7)
                    ind = 2;
                else if (hData[i].MJD < 3.7)
                    ind = 3;
                else if (hData[i].MJD < 4.5)
                    ind = 4;
                else
                    ind = 5;                                
                d2[ind] = d2[ind] + h_dlsq2[i];
                w[ind] = w[ind] + 1;
            }
            double sum = 0.0;
            double W;
            double SW = 0.0;
            for (i=0; i<6; i++)
            {
                if (i == 2 || i == 3)
                    // Giving extra weight to multi-featured regions 2 and 3:
                    W = 2;
                else
                    W = 1;
                sum = sum + W*d2[i]/w[i];
                SW = SW + W;
            }
            double dist = sqrt(sum/SW);
            
            printf("chi2_plot = %13.6e, lsq = %13.6e\n", h_chi2_plot, dist);
            
            #ifndef NOPRINT
            fp = fopen("model.dat", "w");
            for (i=0; i<Nplot; i++)
                fprintf(fp, "%13.6e %13.6e\n", hPlot[i].MJD, h_Vmod[i]);
            fclose(fp);
            
            fp = fopen("data.dat", "w");
            for (i=0; i<N_data; i++)
                fprintf(fp, "%13.6e %13.6e %13.6e w\n", hData[i].MJD, hData[i].V, 1/sqrt(hData[i].w));
            fclose(fp);
            
            fp = fopen("lines.dat", "w");
            for (i=0; i<C_POINTS*BSIZE; i++)
            {
                fprintf(fp, "%13.6e ", 2.0 * DELTA_MAX * ((i+1.0)/(C_POINTS*BSIZE) - 0.5));
                for (int iparam=0; iparam<N_PARAMS; iparam++)
                {
                    CHI_FLOAT xx = h_chi2_lines[iparam][i];
                    if (isnan(xx))
                        xx=1e30;
                    fprintf(fp, "%13.6e ", xx);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
            #endif        
        }        
        
    }
    #endif   // GPU 
    
    // CPU based chi^2:
    //    double chi2_cpu;
    //    chi2(N_data, N_filters, params, &chi2_cpu);    
    
    return 0;  
}
