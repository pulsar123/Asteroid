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
    double params[N_PARAMS];
    CHI_FLOAT chi2_tot=1e32;
    int i;
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
    double P_phi2;
    float hPphi2;
    #endif   
    
    // Observational data:
    int N_data; // Number of data points
    int N_filters; // Number of filters used in the data        
    int Nplot = 0;
    

    // Array describing all optimizable model parameters (initializing only the first segment - i_seg=0)    
    // Set the Frozen value to 1 to fix (exclude from optimization) the corresponding parameters for all segments
    // !!! I need a way to provide the values for the frozen parameters
    // Because of the dependencies, the following order has to be followed: c_tumb -> b_tumb -> Es -> psi_0, and c_tumb -> c -> b, and Es -> L (for P_PHI, P_BOTH)
    int Property[N_PARAMS][N_COLUMNS] = {
        
//  P_*:     type,      independent, frozen, iseg,  multi_segment, periodic
           { T_theta_M, 1,           0,      0,     0,             0},  // theta_M
           { T_phi_M,   1,           0,      0,     0,             1},  // phi_M
           { T_phi_0,   1,           0,      0,     0,             1},  // phi_0
    #ifdef TREND                                  
           { T_A,       1,           0,      0,     1,             0},  // A
    #endif
    #ifdef TORQUE
           { T_theta_K, 1,           0,      0,     0,             0},  // theta_K
           { T_phi_K,   1,           0,      0,     0,             1},  // phi_K
           { T_phi_F,   1,           0,      0,     0,             1},  // phi_F
           { T_K,       1,           0,      0,     0,             0},  // K
    #endif
           { T_c_tumb,  1,           0,      0,     1,             0},  // c_tumb
           { T_b_tumb,  0,           0,      0,     1,             0},  // b_tumb
           { T_Es,      0,           0,      0,     0,             0},  // Es
           { T_L,       1,           0,      0,     0,             0},  // L / P_psi / P_phi
           { T_psi_0,   0,           0,      0,     0,             0},  // psi_0
    #ifdef BC                                  
           { T_c,       1,           0,      0,     1,             0},  // c
           { T_b,       0,           0,      0,     1,             0}   // b
    #endif                                  
    
    };
    

    #ifdef SEGMENT
    // Adding parameters for other data segments (i_seg>0) - only those which are not Multi-segment
    int j0 = N_PARAMS0 - 1;
    for (int i_seg=1; i_seg<N_SEG; i_seg++)
    {
        for (int j=0; j<N_PARAMS0; j++)
            // Only Multi-segment=0 parameters are copied:
            if (Property[j][P_multi_segment] == 0)
            {
                j0++;
                for (int k=0; k<N_COLUMNS; k++)
                    Property[j0][k] = Property[j][k];
                // Assigning the correct iseg value to segments:
                Property[j0][P_iseg] = i_seg;
            }
    }
    // Sanity check:
    if (j0 != N_PARAMS - 1)
    {
        printf ("j0 != N_PARAMS!\n");
        exit(1);
    }
    #endif                                
    
    // Initializing the Types vector (contains iparams for each type,iseg combo):    
    int Types[N_TYPES][N_SEG];
    for (j=0; j<N_TYPES; j++)
        for (int iseg=0; iseg<N_SEG; iseg++)
            Types[j][iseg] = -1;
    for (i=0; i<N_PARAMS; i++)
    {
        if (Property[i][P_multi_segment] == 0)
        {
            Types[Property[i][P_type]][Property[i][P_iseg]] = i;
        }
        else
        {
            // For multi-segment parameters, all iseg columns get the same i value:
            for (int iseg=0; iseg<N_SEG; iseg++)
                Types[Property[i][P_type]][iseg] = i;
        }
    }
    
    
    
    if (argc == 1)
    {
        printf("\n Command line arguments:\n\n");
        printf("-i name  : input (data) file name\n");
        printf("-o name  : output (results) file name\n");
        printf("-m param1 param2 ... paramN  : input model parameters, for plotting and re-optimization\n");
        printf("-plot    : plotting (only makes sense when -m is also used)\n");
        #if defined(P_PHI) || defined(P_BOTH)
        printf("-Pphi min max  : minimum and maximum values for Pphi period, in hours\n");
        #endif
        #if defined(P_PSI) || defined(P_BOTH)
        printf("-Ppsi min max  : minimum and maximum values for Ppsi period, in hours\n");
        #endif
        printf("-f type_constant value: forces the parameter with the type_constant to be frozen during optimization at \"value\" \n");
        printf("\n");
    }
    
    
    // Processing the command line arguments:
    int j = 1;
    int j_input = -1;
    int j_results = -1;
    int i_frozen = -1;
    int Fi[N_TYPES];
    double Fv[N_TYPES];  
    while (j < argc)
    {
        // Input (data) file name:
        if (strcmp(argv[j], "-i") == 0)
        {
            j_input = j + 1;
            j = j + 2;;
            if (j >= argc)
                break;
        }
        
        // Output (results) file name:
        if (strcmp(argv[j], "-o") == 0)
        {
            j_results = j + 1;
            j = j + 2;
            if (j >= argc)
                break;
        }
        
        #if defined(P_PHI) || defined(P_BOTH)
        // Range of Pphi period (hours)
        if (strcmp(argv[j], "-Pphi") == 0)
        {
            Pphi1 = atof(argv[j+1]);
            Pphi2 = atof(argv[j+2]);
            hPphi = P_phi1 / 24.0 / (2*PI);
            hPphi2 = P_phi2 / 24.0 / (2*PI);    
            j = j + 3;
            if (j >= argc)
                break;
        }
        #endif
        
        #if defined(P_PSI) || defined(P_BOTH)
        // Range of Ppsi period (hours)
        if (strcmp(argv[j], "-Ppsi") == 0)
        {
            Ppsi1 = atof(argv[j+1]);
            Ppsi2 = atof(argv[j+2]);
            j = j + 3;
            if (j >= argc)
                break;
        }
        #endif
        
        // Input model parameters (for plotting or re-optimization):
        if (strcmp(argv[j], "-m") == 0)
        {
            for (int k=0; k<N_PARAMS; k++)
            {
                params[k] = atof(argv[j+1+k]);
                if (Property[k][P_type] == T_L)
                    params[k] = 48.0*PI/params[k];
            }
            j = j + 1 + N_PARAMS;
            if (j >= argc)
                break;
        }
        
        // the code only does plotting:
        if (strcmp(argv[j], "-plot") == 0)
        {
            Nplot = NPLOT;
            j = j + 1;
            if (j >= argc)
                break;
        }

        // Frozen parameter constant and the value
        if (strcmp(argv[j], "-f") == 0)
        {
            i_frozen++;
            Fi[i_frozen] = atoi(argv[j+1]);
            Fv[i_frozen] = atof(argv[j+2]);
            j = j + 2;
            if (j >= argc)
                break;
        }
    }
    
    // Total number of frozen parameter types:
    int N_frozen = i_frozen + 1;
    
    Is_GPU_present();
    
    // Reading all input data files, allocating and initializing observational data arrays   
    read_data(argv[j_input], &N_data, &N_filters, Nplot);
    
    int N_threads = N_BLOCKS * BSIZE;
    
    gpu_prepare(N_data, N_filters, N_threads, Nplot);
    
    // Limits for each independent model parameter during optimization:
    CHI_FLOAT hLimits[2][N_TYPES];
    
    // Theta_M (angle between barycentric Z axis and angular momentum vector M); range 0...pi
    hLimits[0][T_theta_M] = 0.001/RAD;
    hLimits[1][T_theta_M] = 179.999/RAD;
    
    // Angular momentum L value, radians/day; if P is period in hours, L=48*pi/P
    hLimits[0][T_L] = 48.0*PI / 10; // 8.5
    hLimits[1][T_L] = 48.0*PI / 0.1; // 0.4    
    #ifndef REOPT
    // In P_PHI mode has a different meaning: 48*pi/Pphi2 ... 48*pi/Pphi1 (used to generate L)
    #ifdef P_PHI
    hLimits[0][T_L] = 48.0*PI / Pphi2;
    hLimits[1][T_L] = 48.0*PI / Pphi1;
    #endif
    // In P_PSI mode has a different meaning: 1/Ppsi2 ... 1/Ppsi1, 1/days (used to derive L)
    #if defined(P_PSI) || defined(P_BOTH)
    hLimits[0][T_L] = 24.0/Ppsi2;
    hLimits[1][T_L] = 24.0/Ppsi1;
    #endif
    #endif    
    
    #ifdef TREND
    //scaling parameter "A" for de-trending the brightness curve, in magnitude/radian units (to be multiplied by the phase angle alpha to get magnitude correction):
    // Physically plausible values are negative, -1.75 ... -0.5 mag/rad
    hLimits[0][T_A] = -10;
    hLimits[1][T_A] = 10;
    #endif        
    
    #ifdef TORQUE
    // First orientation angle for the vector r to the point on the surface where the torque is applied, theta_K; 0 ... 180
    hLimits[0][T_theta_K] = 0.001/RAD;
    hLimits[1][T_theta_K] = 179.999/RAD;
    // Amplitude of the torque, K; >=0; units are 1/day^2
    hLimits[0][T_K] = 0;
    hLimits[1][T_K] = 600;
    #endif        
    
    // c_tumb (physical (tumbling) value of the axis c size; always smallest)
    hLimits[0][T_c_tumb] = log(0.002);
    hLimits[1][T_c_tumb] = log(1.0);                
    
    // b_tumb (physical (tumbling) value of the axis b size; always intermediate), in relative to c_tumb units (between 0: 1, and 1: log(c_tumb))
    // For a symmetric cigar / disk, freeze the limits to 1 / 0
    hLimits[0][T_b_tumb] = 0;
    hLimits[1][T_b_tumb] = 1;
    
    // Es: relative tumbling energy limits; 0...0.5 is for SAM, 0.5...1 is for LAM
    // Both limits frozen at very small value -> simple SAM rotator; frozen close to 1 -> simple LAM rotator
//    hLimits[0][T_Es] = 0;
//    hLimits[1][T_Es] = 1;
    
    #ifdef BC
    // Limits on the geometric c shape parameter = the limits on the dynamic c_tumb parameter:
    hLimits[0][T_c] = hLimits[0][T_c_tumb];
    hLimits[1][T_c] = hLimits[1][T_c_tumb];

    // Limits on the geometric b shape parameter, in relative to c units (between 0: 1, and 1: log(c_tumb))
    // For a symmetric cigar / disk, freeze the limits to 1 / 0
    hLimits[0][T_b] = 0;
    hLimits[1][T_b] = 1;
    #endif
    
    // Updating hLimits and Property for all frozen parameters:
    for (int i_frozen=0; i<N_frozen; i_frozen++)
    {
        int itype = Ni[i_frozen];
        double value = Nv[i_frozen];
        // Loop over params from all segments:
        for (i=0; i<N_PARAMS; i++)
            if(Property[i][P_type] == itype)
            {
                Property[i][P_frozen] = 1;
                // Setting both limits to identical values = value from the command line argument
                // This overrides any static Limits definitions from above
                hLimits[0][itype] = value;
                hLimits[1][itype] = value;
            }
    }
    
    ERR(cudaMemcpyToSymbol(dProperty, Property, N_COLUMNS*N_PARAMS*sizeof(int), 0, cudaMemcpyHostToDevice));                
    ERR(cudaMemcpyToSymbol(dTypes, Types, N_TYPES*N_SEG*sizeof(int), 0, cudaMemcpyHostToDevice));                
    ERR(cudaMemcpyToSymbol(dLimits, hLimits, 2*N_TYPES*sizeof(CHI_FLOAT), 0, cudaMemcpyHostToDevice));                
    #ifdef P_BOTH
    ERR(cudaMemcpyToSymbol(dPphi, &hPphi, sizeof(float), 0, cudaMemcpyHostToDevice));                
    ERR(cudaMemcpyToSymbol(dPphi2, &hPphi2, sizeof(float), 0, cudaMemcpyHostToDevice));                
    #endif    
    
    #ifdef NUDGE
    prepare_chi2_params(&N_data);
    #endif
    struct x2_struct x2_params;
    #ifdef P_BOTH
    x2_params.Pphi = hPphi;
    #ifdef PHI2
    x2_params.Pphi2 = hPphi2;
    #endif
    #endif
    
    
    if (Nplot == 0)
    {
        printf("\n*** Simplex optimization ***\n\n");
        printf("  N_threads = %d\n", N_threads);        
        
        // Initializing the device random number generator:
        curandState* d_states;
        ERR(cudaMalloc ( &d_states, N_BLOCKS*BSIZE*sizeof( curandState ) ));
        // setup seeds, initialize d_f
        #ifdef TIMING        
        setup_kernel <<< N_BLOCKS, BSIZE >>> ( d_states, (unsigned long)0, d_f);
        #else
        setup_kernel <<< N_BLOCKS, BSIZE >>> ( d_states, (unsigned long)(time(NULL)), d_f);
        #endif
        
        #ifdef REOPT
        ERR(cudaMemcpyToSymbol(d_params0, &params, N_PARAMS*sizeof(double), 0, cudaMemcpyHostToDevice));
        #endif        
        
        ERR(cudaDeviceSynchronize());    
        
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
            chi2_gpu<<<N_BLOCKS, BSIZE>>>(dData, N_data, N_filters, d_states, d_f, d_params, x2_params);
            
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
                fp = fopen(argv[j_results], "w");
                for (i=0; i<N_BLOCKS; i++)
                {
                    fprintf(fp,"%13.6e ",  h_f[i]);
                    for (j=0; j<N_PARAMS; j++)
                        if (j == T_L)
                            fprintf(fp,"%15.11f ",  48*PI/params[i][j]);
                            else
                            fprintf(fp,"%15.11f ",  params[i][j]);
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
                printf("%13.6e ",  h_f[i_best]);
                for (j=0; j<N_PARAMS; j++)
                    if (j == T_L)
                        printf("%15.11f ",  48*PI/params[i_best][j]);
                    else
                        printf("%15.11f ",  params[i_best][j]);
                    printf("\n");
                fflush(stdout);
            }
            
            ERR(cudaDeviceSynchronize());
            
            ERR(cudaMemcpy(h_f, d_f, N_BLOCKS * sizeof(CHI_FLOAT), cudaMemcpyDeviceToHost));
            ERR(cudaMemcpy(h_params, d_params, N_BLOCKS * N_PARAMS * sizeof(double), cudaMemcpyDeviceToHost));
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
        ERR(cudaMemcpyFromSymbol(&h_delta_V0, d_delta_V0, sizeof(CHI_FLOAT), 0, cudaMemcpyDeviceToHost));
        FILE * fV=fopen("delta_V","w");
        fprintf(fV, "%8.4f\n", h_delta_V0);
        fclose(fV);
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
            // The time here is corrected for light travel
            fprintf(fp, "%13.7f %13.6e\n", hMJD0+hPlot[i].MJD, h_Vmod[i]);
        fclose(fp);
        
        fp = fopen("data.dat", "w");
        for (i=0; i<N_data; i++)
            // The time here is corrected for light travel
            fprintf(fp, "%13.7f %13.6e %13.6e w\n", hMJD0+hData[i].MJD, hData[i].V, 1/sqrt(hData[i].w));
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
    
    
    
    return 0;  
}
