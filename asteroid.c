/* CUDA code to fit physical models of a tumbing asteroid with optional torque to light curve data.
 * 
 * Paper:  
 * 
  Written by Sergey Mashchenko, 2017-2019
*/

#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curand_kernel.h>
#define MAIN
#include "asteroid.h"

int main (int argc,char **argv)
{
    FILE *fp;
    double params[N_PARAMS];
    CHI_FLOAT chi2_tot=1e32;
    int i;
    #if defined(P_PHI) || defined(P_BOTH)
    double Pphi1, Pphi2;
    #endif
    #ifdef P_PSI
    float hPphi;
    float hPphi2;
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
    int travel = 0;
    int keep = 0;
    int best = 0;
    int reopt = 0;
    int inp_model = 0;
    #ifdef PROFILES    
    float dx_rand = DELTA_MAX;
    #else    
    float dx_rand = DX_RAND;
    #endif
    #ifdef ANIMATE
    int i1_input = 0;
    int i2_input = NPLOT;
    #endif
    
    // Observational data:
    int N_data; // Number of data points
    int N_filters; // Number of filters used in the data        
    int Nplot = 0;
    unsigned long int seed = 0;
    int Ncases = -1;
    int Nstages = 1;
    int model = 0;
    #ifdef MINIMA_TEST
    int has_delta_V = 0;
    CHI_FLOAT delta_V = 0.0;
    #endif
    
    #ifdef ONE_LE
    int const LE = 1;
    #else
    int const LE = 0;
    #endif
    

    /* Array describing all optimizable model parameters (initializing only the first segment - i_seg=0)    
       Set the Frozen value to 1 to fix (exclude from optimization) the corresponding parameters for all segments
       Because of the dependencies, the following order has to be followed: c_tumb -> b_tumb -> Es -> psi_0, and c_tumb -> c -> b, and Es -> L (for P_PHI, P_BOTH)
       P_periodic values:
        PERIODIC: always periodic, 2*pi period
        PERIODIC_LAM: only periodic when LAM (psi_0)
        SOFT_BOTH: both limits are soft (total allowed extended range -infinity ... infinity in dimensionless units)
        HARD_LEFT: left limit (0 in dimensionless units) is hard
        HARD RIGHT: right limit (1 in dimensionless units) is hard
        HARD_BOTH: both limits are hard (0 ... 1 in dimensionless units)
       */
    int Property[N_PARAMS][N_COLUMNS] = {
        
//  P_*:     type,      independent, frozen, iseg,  multi_segment, periodic
           { T_theta_M, 1,           0,      0,     0,     HARD_BOTH},  // theta_M
           { T_phi_M,   1,           0,      0,     0,      PERIODIC},  // phi_M
           { T_phi_0,   1,           0,      0,     0,      PERIODIC},  // phi_0
    #ifdef TREND                                  
           { T_A,       1,           0,      0,     1,     HARD_BOTH},  // A
    #endif
    #ifdef TORQUE
           { T_Ti,      1,           0,      0,     0,     SOFT_BOTH},  // Ti - Tb
           { T_Ts,      1,           0,      0,     0,     SOFT_BOTH},  // Ts - Tc
           { T_Tl,      1,           0,      0,     0,     SOFT_BOTH},  // Tl - Ta
    #endif
    #ifdef TORQUE2
           { T_T2i,     1,           0,      0,     0,     SOFT_BOTH},  // T2i
           { T_T2s,     1,           0,      0,     0,     SOFT_BOTH},  // T2s
           { T_T2l,     1,           0,      0,     0,     SOFT_BOTH},  // T2l
           { T_Tt,      1,           0,      0,     0,     HARD_BOTH},  // Tt
    #endif
           { T_c_tumb,  1,           0,      0,     1,    HARD_RIGHT},  // c_tumb
           { T_b_tumb,  0,           0,      0,     1,     HARD_BOTH},  // b_tumb
           { T_Es,      0,           0,      0,    LE,     HARD_BOTH},  // Es
           { T_L,       1,           0,      0,    LE,     HARD_LEFT},  // L
           { T_psi_0,   0,           0,      0,     0,  PERIODIC_LAM},  // psi_0
    #ifdef BC                                  
           { T_c,       1,           0,      0,     1,    HARD_RIGHT},  // c
           { T_b,       0,           0,      0,     1,     HARD_BOTH},  // b
    #endif                                  
    #if defined(ROTATE) || defined(BW_BALL)
           { T_theta_R, 1,           0,      0,     1,     HARD_BOTH},  // theta_R (polar angle theta under BW_BALL)
           { T_phi_R,   1,           0,      0,     1,      PERIODIC},  // phi_R (azimuthal angle under BW_BALL is phi_R-90 dgr.)
    #endif
    #ifdef ROTATE
           { T_psi_R,   1,           0,      0,     1,      PERIODIC},  // psi_R
    #endif
    #ifdef BW_BALL                                
           { T_kappa,   1,           0,      0,     1,     SOFT_BOTH},  // kappa
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
    for (int j=0; j<N_TYPES; j++)
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
        printf("-best : only keep the best result\n");
        #ifdef MINIMA_TEST
//        printf("-delta_V value : delta_V value, only in MINIMA_TEST mode\n");
        #endif
        #ifdef RMSD
        printf("-dx value : maximum shift for parameters in scale-free units (0...1)\n");
        #endif
        printf("-f type_constant value: forces the parameter with the type_constant to be frozen during optimization at \"value\" \n");
        printf("-i name : input (data) file name\n");
        #ifdef ANIMATE        
        printf("-i1 index : index of the first snapshot\n");
        printf("-i2 index : index of the last snapshot minus one\n");
        #endif
        printf("-keep : keep all intermediate results, not just the best ones\n");
        printf("-l type_constant limit1 limit2 : specify range for the parameter type_constant\n");
        printf("-m param1 param2 ... paramN : input model parameters, for plotting and re-optimization\n");
        printf("     If one of the parameters has a special value of \"v\", it is allowed to vary randomly within its full range.\n");
        printf("-N number : exit after \"number\" cycles\n");
        printf("-Nstages number : each initial optimization stage is followed by \"number-1\" reoptimization stages\n");
        printf("-o name : output (results) file name\n");        
        printf("-plot : plotting (only makes sense when -m is also used)\n");
        #if defined(P_PHI) || defined(P_BOTH)
        printf("-Pphi min max : minimum and maximum values for Pphi period, in hours\n");
        #endif
        #if defined(P_PSI) || defined(P_BOTH)
        printf("-Ppsi min max : minimum and maximum values for Ppsi period, in hours\n");
        #endif
        printf("-reopt : reoptimize the model provided with -m switch\n");
        printf("-seed SEED : use the SEED number to initialize the random number generator\n");
        printf("-t : travelling reoptimization\n");
        printf("\n");
        exit(0);
    }

    if (argc == 1)
        exit(0);
    
    // Processing the command line arguments:
    int j = 1;
    int j_input = -1;
    int j_results = -1;
    int i_frozen = -1;
    int i_limits = -1;
    int Fi[N_TYPES], Li[N_TYPES];
    double Fv[N_TYPES], L0[N_TYPES], L1[N_TYPES];  
    while (j < argc)
    {
        if (argv[j][0] != '-')
        {
            printf("Bad arguments!\n");
            exit(1);
        }
        
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
        
        #ifdef RMSD
        if (strcmp(argv[j], "-dx") == 0)
        {
            dx_rand = atof(argv[j+1]);
            j = j + 2;
            if (j >= argc)
                break;
        }
        #endif
        
        #ifdef ANIMATE
        if (strcmp(argv[j], "-i1") == 0)
        {
            i1_input = atoi(argv[j+1]);
            j = j + 2;
            if (j >= argc)
                break;
        }
        if (strcmp(argv[j], "-i2") == 0)
        {
            i2_input = atoi(argv[j+1]);
            j = j + 2;
            if (j >= argc)
                break;
        }
        #endif
        
        // Input model parameters (for plotting or re-optimization):
        if (strcmp(argv[j], "-m") == 0)
        {
            for (int k=0; k<N_PARAMS; k++)
            {
                if (j+1+k == argc)
                {
                    printf("Not enough of model parameters in -m switch!\n");
                    exit(1);
                }
                if (argv[j+1+k][0] == 'v')
                {
                    // The special value of "-1" for P_frozen parameter means this parameter is randomly changing within its full range:
                    Property[k][P_frozen] = -1;
                    params[k] = 0;  // Arbitrary value, not used
                }
                else
                {
                    params[k] = atof(argv[j+1+k]);
#ifdef MY_L
                    if (Property[k][P_type] == T_L)
                        params[k] = 48.0*PI/params[k];
#endif
                }
            }
            j = j + 1 + N_PARAMS;
            model = 1;
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

        // traveling reoptimization:
        if (strcmp(argv[j], "-t") == 0)
        {
            travel = 1;
            j = j + 1;
            if (j >= argc)
                break;
        }

        if (strcmp(argv[j], "-reopt") == 0)
        {
            reopt = 1;
            j = j + 1;
            if (j >= argc)
                break;
        }

        if (strcmp(argv[j], "-keep") == 0)
        {
            keep = 1;
            j = j + 1;
            if (j >= argc)
                break;
        }

        if (strcmp(argv[j], "-best") == 0)
        {
            best = 1;
            j = j + 1;
            if (j >= argc)
                break;
        }

        if (strcmp(argv[j], "-seed") == 0)
        {
            seed = strtoul(argv[j+1], NULL, 10);
            j = j + 2;
            if (j >= argc)
                break;
        }

        if (strcmp(argv[j], "-N") == 0)
        {
            Ncases = atoi(argv[j+1]);
            j = j + 2;
            if (j >= argc)
                break;
        }

        if (strcmp(argv[j], "-Nstages") == 0)
        {
            Nstages = atoi(argv[j+1]);
            j = j + 2;
            if (j >= argc)
                break;
        }

        // Frozen parameter constant and the value
        if (strcmp(argv[j], "-f") == 0)
        {
            i_frozen++;
            Fi[i_frozen] = atoi(argv[j+1]);
            Fv[i_frozen] = atof(argv[j+2]);
            j = j + 3;
            if (j >= argc)
                break;
        }

        // Parameter constant and the two limits (full range)
        if (strcmp(argv[j], "-l") == 0)
        {
            i_limits++;
            Li[i_limits] = atoi(argv[j+1]);
            L0[i_limits] = atof(argv[j+2]);
            L1[i_limits] = atof(argv[j+3]);
            j = j + 4;
            if (j >= argc)
                break;

            
        }
        
        #ifdef MINIMA_TEST        
        if (strcmp(argv[j], "-delta_V") == 0)
        {
            delta_V = atof(argv[j+1]);
            has_delta_V = 1;
            j = j + 2;
            if (j >= argc)
                break;
        }
        #endif
        
    }  // while argc loop
    
    if (j_input == -1)
      {
          printf("-i parameter is missing!\n");
          exit(1);
      }
    if ((reopt || Nplot>0) && !model)
    {
        printf("-reopt and -plot switches require -m switch!\n");
        exit(1);
    }
    if (reopt==0 & Nplot==0 && model)
    {
        printf("-m can only be used together with -reopt or -plot switches!\n");
        exit(1);
    }
    #ifdef MINIMA_TEST        
    /*
    if (has_delta_V == 0)
    {
        printf("-delta_V switch is required in MINIMA_TEST mode!\n");
        exit(1);
    }
    */
    #endif

        // Total number of frozen parameter types:
    int N_frozen = i_frozen + 1;
    // Total number of parameters with custom limits:
    int N_limits = i_limits + 1;
    
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
    /*
    if (!reopt)
    {
        // In P_PHI mode storing 48*pi/Pphi2 ... 48*pi/Pphi1 (used to generate L)
        #ifdef P_PHI
        hLimits[0][T_L] = 48.0*PI / Pphi2;
        hLimits[1][T_L] = 48.0*PI / Pphi1;
        #endif
        // In P_PSI mode has a different meaning: 1/Ppsi2 ... 1/Ppsi1, 1/days (used to derive L)
        #if defined(P_PSI) || defined(P_BOTH)
        hLimits[0][T_L] = 24.0/Ppsi2;
        hLimits[1][T_L] = 24.0/Ppsi1;
        #endif
    }
    */
    
    #ifdef TREND
    //scaling parameter "A" for de-trending the brightness curve, in magnitude/radian units (to be multiplied by the phase angle alpha to get magnitude correction):
    // Physically plausible values are negative, -1.75 ... -0.5 mag/rad
    hLimits[0][T_A] = -5;
    hLimits[1][T_A] = 5;
    #endif        
    
    #ifdef TORQUE
    // Maximum amplitude for torque parameters (units are rad/day)
    double Tmax = 10.0;
    hLimits[0][T_Ti] = -Tmax;
    hLimits[1][T_Ti] =  Tmax;
    hLimits[0][T_Ts] = -Tmax;
    hLimits[1][T_Ts] =  Tmax;
    hLimits[0][T_Tl] = -Tmax;
    hLimits[1][T_Tl] =  Tmax;
    #endif        
    #ifdef TORQUE2
    hLimits[0][T_T2i] = -Tmax;
    hLimits[1][T_T2i] =  Tmax;
    hLimits[0][T_T2s] = -Tmax;
    hLimits[1][T_T2s] =  Tmax;
    hLimits[0][T_T2l] = -Tmax;
    hLimits[1][T_T2l] =  Tmax;
    // The fractional split point (in time) between the two torque regimes; 0 is for torque2 only, 1 is for torque1 only
    hLimits[0][T_Tt]  =  0.001;
    hLimits[1][T_Tt]  =  0.999;
    #endif        
    
    // c_tumb (physical (tumbling) value of the axis c size; always smallest)
    hLimits[0][T_c_tumb] = log(0.01);
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
//    hLimits[0][T_c] = log(0.1);
//    hLimits[1][T_c] = log(1.0);
    hLimits[0][T_c] = hLimits[0][T_c_tumb];
    hLimits[1][T_c] = hLimits[1][T_c_tumb];

    // Limits on the geometric b shape parameter, in relative to c units (between 0: 1, and 1: log(c_tumb))
    // For a symmetric cigar / disk, freeze the limits to 1 / 0
    hLimits[0][T_b] = 0;
    hLimits[1][T_b] = 1;
    #endif

    #if defined(ROTATE) || defined(BW_BALL)
    // Theta_R (polar angle when rotating the brightness ellipsoid relative to the kinematic ellipsoid); range 0...pi
    hLimits[0][T_theta_R] = 0.001/RAD;
    hLimits[1][T_theta_R] = 179.999/RAD;
    #endif
    
    #ifdef BW_BALL
    // dark/bright sides albedo ratio kappa=0...1, log scale
    hLimits[0][T_kappa] = log(0.01);
    hLimits[1][T_kappa] = log(0.1);
    #endif

    // Updating hLimits and Property for all frozen parameters:
    for (int i_frozen=0; i_frozen<N_frozen; i_frozen++)
    {
        int itype = Fi[i_frozen];
        double value = Fv[i_frozen];
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

    // Applying custom limits from the commnd line arguments:
    for (int i_limits=0; i_limits<N_limits; i_limits++)
    {
        int itype = Li[i_limits];
        hLimits[0][itype] = L0[i_limits];
        hLimits[1][itype] = L1[i_limits];
    }
 
    struct x2_struct x2_params;
    #ifdef P_BOTH
    x2_params.Pphi = Pphi1 / 24.0 / (2*PI);
      #ifdef PHI2
      x2_params.Pphi2 = Pphi2 / 24.0 / (2*PI);
      #endif
    #endif
    #if defined(P_PSI) || defined(P_BOTH)
    x2_params.Ppsi1 = 24.0/Ppsi2;
    x2_params.Ppsi2 = 24.0/Ppsi1;
    #endif
     
    ERR(cudaMemcpyToSymbol(dProperty, Property, N_COLUMNS*N_PARAMS*sizeof(int), 0, cudaMemcpyHostToDevice));                
    ERR(cudaMemcpyToSymbol(dTypes, Types, N_TYPES*N_SEG*sizeof(int), 0, cudaMemcpyHostToDevice));                
    ERR(cudaMemcpyToSymbol(dLimits, hLimits, 2*N_TYPES*sizeof(CHI_FLOAT), 0, cudaMemcpyHostToDevice));                
    ERR(cudaMemcpyToSymbol(d_x2_params, &x2_params, sizeof(struct x2_struct), 0, cudaMemcpyHostToDevice));                
    
    #ifdef NUDGE
    prepare_chi2_params(&N_data);
    #endif
   
    
    if (Nplot == 0)                
    {
        #ifndef ANIMATE
        printf("\n*** Simplex optimization ***\n\n");
        printf("  N_threads = %d\n", N_threads);        
        
        if (j_results == -1)
        {
            printf("Bad output file name!\n");
            exit(1);
        }
        
        // Initializing the device random number generator:
        curandState* d_states;
        ERR(cudaMalloc ( &d_states, N_BLOCKS*BSIZE*sizeof( curandState ) ));
        // setup seeds, initialize d_f
        if (seed == 0)
            // seed=0 when no seed was provided on the command line; using time to randomize it:
            setup_kernel <<< N_BLOCKS, BSIZE >>> ( d_states, (unsigned long)(time(NULL)), d_f, 1);
        else
            // Otherwise use the explicitely provided value of seed (good for post-processing, profiling and debugging):
            setup_kernel <<< N_BLOCKS, BSIZE >>> ( d_states, seed, d_f, 1);
        
        if (reopt)
            ERR(cudaMemcpyToSymbol(d_params0, params, N_PARAMS*sizeof(double), 0, cudaMemcpyHostToDevice));
        
        ERR(cudaDeviceSynchronize());    
        
        int loop_counter = 0;
        
        #ifdef RMSD
        float PAR_min[N_PARAMS], PAR_max[N_PARAMS];
        for (int j=0; j<N_PARAMS; j++)
        {
            PAR_min[j] =  1e30;
            PAR_max[j] = -1e30;
        }
        printf("\n");
        fp = fopen(argv[j_results], "w");
        #endif
        
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
            
            // The kernel:
            #ifdef RMSD
            chi2_gpu_rms<<<N_BLOCKS, BSIZE>>>(dData, N_data, N_filters, reopt, Nstages, d_states, d_f, d_params, d_dV, dx_rand, dpar_min, dpar_max);
            #else
            chi2_gpu<<<N_BLOCKS, BSIZE>>>(dData, N_data, N_filters, reopt, Nstages, d_states, d_f, d_params, d_dV);
            #endif
            
            #ifdef TIMING
            cudaEventRecord(stop, 0);
            cudaEventSynchronize (stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            printf("GPU time: %.2f ms\n", elapsed);
            exit(0);
            #endif        
            
            ERR(cudaDeviceSynchronize());
            
            #ifdef RMSD  // RMSD mode (finding confidence intervals for the input model)

            ERR(cudaMemcpy(hpar_min, dpar_min, N_BLOCKS * N_PARAMS * sizeof(float), cudaMemcpyDeviceToHost));
            ERR(cudaMemcpy(hpar_max, dpar_max, N_BLOCKS * N_PARAMS * sizeof(float), cudaMemcpyDeviceToHost));
            int h_Ntot, h_Nbad;
            ERR(cudaMemcpyFromSymbol(&h_Ntot, d_Ntot, sizeof(int), 0, cudaMemcpyDeviceToHost));
            ERR(cudaMemcpyFromSymbol(&h_Nbad, d_Nbad, sizeof(int), 0, cudaMemcpyDeviceToHost));
            if (loop_counter == 1)
            {
                float h_f0, h_f1;            
                ERR(cudaMemcpyFromSymbol(&h_f0, d_f0, sizeof(float), 0, cudaMemcpyDeviceToHost));
                ERR(cudaMemcpyFromSymbol(&h_f1, d_f1, sizeof(float), 0, cudaMemcpyDeviceToHost));
                printf("f0=%f, f1=%f\n\n", h_f0, h_f1);
            }
            printf("%e ", h_Nbad/(double)h_Ntot);

            for (int j=0; j<N_PARAMS; j++)
            {
                for (int i=0; i<N_BLOCKS; i++)
                {
                    int k = i*N_PARAMS + j;
                    if (hpar_min[k] < PAR_min[j])
                        PAR_min[j] = hpar_min[k];
                    if (hpar_max[k] > PAR_max[j])
                        PAR_max[j] = hpar_max[k];
                }
                // Printing the cumulative values of the parameters' confidence intervals:
                // Warning - MY_L is ignored here: proper L values are always printed
                printf("%13.6e %13.6e ; ", PAR_min[j], PAR_max[j]);
                fprintf(fp, "%13.6e %13.6e ; ", PAR_min[j], PAR_max[j]);
            }
            printf("\n");
            fflush(stdout);
            fprintf(fp, "\n");
            fflush(fp);
           
            #else  // Normal mode (global optimization):
            
            ERR(cudaMemcpy(h_params, d_params, N_BLOCKS * N_PARAMS * sizeof(double), cudaMemcpyDeviceToHost));
            ERR(cudaMemcpy(h_dV, d_dV, N_BLOCKS * N_FILTERS * sizeof(double), cudaMemcpyDeviceToHost));
            ERR(cudaMemcpy(h_f, d_f, N_BLOCKS * sizeof(CHI_FLOAT), cudaMemcpyDeviceToHost));
            ERR(cudaDeviceSynchronize());

            if (loop_counter > 0)
            {
                // Finding the best result between all threads:        
                int i_best = 0;
                chi2_tot = 1e32;
                int Nresults = 0;
                double iii;
                int l = -1;
                for (i=0; i<N_BLOCKS; i++)
                {
                    if (h_f[i] < 1e29)
                        Nresults++;
                    if (h_f[i] < chi2_tot)
                    {
                        chi2_tot = h_f[i];
                        i_best = i;
                    }
                    // Bringing periodic parameters to the canonic range of values
                    for (j=0; j<N_PARAMS; j++)
                    {
                        l = l++;                        
                        if (Property[j][P_periodic] == 1 || Property[j][P_type] == T_psi_0)
                            // To [0,2*pi[ range:
                            h_params[l] = 2*PI * modf(h_params[l]/(2*PI), &iii);
                    }
                    
                }
                
                if (reopt && travel)
                    // In "traveling reoptimization" mode, we move the starting point to the best point found in the previous cycle
                {
                    for (j=0; j<N_PARAMS; j++)
                        params[j] = h_params[i_best*N_PARAMS + j];
                    ERR(cudaMemcpyToSymbol(d_params0, params, N_PARAMS*sizeof(double), 0, cudaMemcpyHostToDevice));
                }

                // Priting the best result:
                printf("%13.6e ",  h_f[i_best]);
                for (j=0; j<N_PARAMS; j++)
#ifdef MY_L                    
                    if (Property[j][P_type] == T_L)
                        printf("%15.11f ",  48*PI/h_params[i_best*N_PARAMS + j]);
                    else
#endif
                        printf("%15.11f ",  h_params[i_best*N_PARAMS + j]);
                    printf("\n");
                fflush(stdout);

                if (keep)
                    // In the "keep" mode, we append new results
                    fp = fopen(argv[j_results], "a");
                    else
                    // In the default mode, we overwtite the results with the current best ones
                    fp = fopen(argv[j_results], "w");
                int i1, i2;
                if (best)
                    // If best=1, printing only the best model (one line)
                {
                    i1 = i_best;   i2 = i_best + 1;
                }
                else
                    // If best <> 1, printing all models
                {
                    i1 = 0;  i2 = N_BLOCKS;
                }
                for (i=i1; i<i2; i++)
                {
                    fprintf(fp,"%13.6e ",  h_f[i]);
                    for (int m=0; m<N_filters; m++)
                        fprintf(fp,"%13.6e ",  h_dV[i*N_FILTERS + m]);
                    for (j=0; j<N_PARAMS; j++)
#ifdef MY_L                    
                        if (Property[j][P_type] == T_L)
                            fprintf(fp,"%15.11f ",  48*PI/h_params[i*N_PARAMS + j]);
                            else
#endif
                            fprintf(fp,"%15.11f ",  h_params[i*N_PARAMS + j]);
                    fprintf(fp,"\n");
                }
                fclose(fp);
                                
            }  // if loop_counter > 0
            
            if (keep)
                // If we are keeping all intermediate results, we have to reset d_f to 1e30 at the end of each loop:
                setup_kernel <<< N_BLOCKS, BSIZE >>> ( d_states, (unsigned long)0, d_f, 0);

            #endif  // RMSD
            
            if (loop_counter == Ncases)
                break;            
            
        }  // End of the while loop

        #ifdef RMSD
        fclose(fp);
        #endif
    #endif    // if not ANIMATE
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
        
        #ifdef MINIMA_TEST
        minima_test(N_data, N_filters, Nplot, params, Types, delta_V);
        exit(0);
        #endif        
        
        #if defined PROFILES        
        dim3 NB(C_POINTS, N_PARAMS);
        #elif defined ANIMATE
        int Npix = SIZE_PIX*SIZE_PIX;
        int Nblocks = Npix/BSIZE + (Npix%BSIZE>0);
        dim3 NB (Nblocks, 1);
        #else        
        dim3 NB (1, 1);
        #endif        

        ERR(cudaMemcpyToSymbol(d_params0, params, N_PARAMS*sizeof(double), 0, cudaMemcpyHostToDevice));
        
        
        #ifdef ANIMATE
        int h_i1, h_i2;
        
        for (h_i1=i1_input; h_i1<i2_input; h_i1=h_i1+dNPLOT)
        {
            h_i2 = h_i1 + dNPLOT;
            if (h_i2 > NPLOT)
                h_i2 = NPLOT;
            ERR(cudaMemcpyToSymbol(d_i1, &h_i1, sizeof(int), 0, cudaMemcpyHostToDevice));
            ERR(cudaMemcpyToSymbol(d_i2, &h_i2, sizeof(int), 0, cudaMemcpyHostToDevice));
            chi2_plot<<<NB, BSIZE>>>(dData, N_data, N_filters, dPlot, Nplot, d_dlsq2, d_rgb, dx_rand);
            ERR(cudaMemcpy(h_rgb, d_rgb, (long int)dNPLOT * (long int)SIZE_PIX * (long int)SIZE_PIX * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));        
            write_PNGs(h_rgb, h_i1, h_i2);
        }
        return 0;
        
        #else
        chi2_plot<<<NB, BSIZE>>>(dData, N_data, N_filters, dPlot, Nplot, d_dlsq2, dx_rand);        
        #endif
        
        ERR(cudaDeviceSynchronize());
        ERR(cudaMemcpyFromSymbol(&h_Vmod, d_Vmod, Nplot*sizeof(double), 0, cudaMemcpyDeviceToHost));
        ERR(cudaMemcpyFromSymbol(&h_delta_V, d_delta_V, N_FILTERS*sizeof(CHI_FLOAT), 0, cudaMemcpyDeviceToHost));
        FILE * fV=fopen("delta_V","w");
        for (int m=0; m<N_filters; m++)
            fprintf(fV, "%8.4f\n", h_delta_V[m]);
        fclose(fV);
        ERR(cudaMemcpyFromSymbol(&h_chi2_plot, d_chi2_plot, sizeof(CHI_FLOAT), 0, cudaMemcpyDeviceToHost));
        #ifdef PROFILES        
        ERR(cudaMemcpyFromSymbol(&h_chi2_lines, d_chi2_lines, sizeof(h_chi2_lines), 0, cudaMemcpyDeviceToHost));
        ERR(cudaMemcpyFromSymbol(&h_param_lines, d_param_lines, sizeof(h_param_lines), 0, cudaMemcpyDeviceToHost));
        #endif        
        #ifdef PLOT_OMEGA
            ERR(cudaMemcpyFromSymbol(&h_Omega, d_Omega, 6*Nplot*sizeof(double), 0, cudaMemcpyDeviceToHost));
            FILE * fO=fopen("omega.dat","w");
            for (i=0; i<Nplot; i++)
                // K here is converted to period in hours
                fprintf(fO, "%13.7f %16.9e %16.9f %16.9f %16.9e %16.9f %16.9f\n", hMJD0+hPlot[i].MJD, 48*PI/h_Omega[0][i], h_Omega[1][i]*RAD, h_Omega[2][i]*RAD, 48*PI/h_Omega[3][i], h_Omega[4][i]*RAD, h_Omega[5][i]*RAD);
            fclose(fO);
        #endif
          // Printing the initial and final (only for TORQUE) model state (for later computations of P_psi, P_phi etc)
          FILE *fmod=fopen("model_params.dat","w");
          // Initial model:
                    for (j=0; j<N_PARAMS; j++)
                    {
                        // Skipping the torque parameters:
                        #ifdef TORQUE
                        if (Property[j][P_type]==T_Ti || Property[j][P_type]==T_Ts || Property[j][P_type]==T_Tl)
                            continue;
                        #endif
                        #ifdef BC
                        if (Property[j][P_type]==T_c || Property[j][P_type]==T_b)
                            continue;
                        #endif
                        #ifdef TREND
                        if (Property[j][P_type]==T_A)
                            continue;
                        #endif
                        #ifdef BW_BALL
                        if (Property[j][P_type]==T_theta_R || Property[j][P_type]==T_phi_R ||Property[j][P_type]==T_kappa)
                            continue;
                        #endif                        
#ifdef MY_L                    
                        if (Property[j][P_type] == T_L)
                            fprintf(fmod, "%15.11f ",  48*PI/params[j]);
                            else
#endif                                
                            fprintf(fmod, "%15.11f ",  params[j]);
                    }
                    fprintf(fmod, "\n");
        #ifdef TORQUE
          #ifdef LAST
          double L_last, E_last;
          ERR(cudaMemcpyFromSymbol(&L_last, d_L_last, sizeof(double), 0, cudaMemcpyDeviceToHost));
          ERR(cudaMemcpyFromSymbol(&E_last, d_E_last, sizeof(double), 0, cudaMemcpyDeviceToHost));
          // Final model:
                    for (j=0; j<N_PARAMS; j++)
                    {
                        // Skipping the torque parameters:
                        if (Property[j][P_type]==T_Ti || Property[j][P_type]==T_Ts || Property[j][P_type]==T_Tl)
                            continue;
                        #ifdef BC
                        if (Property[j][P_type]==T_c || Property[j][P_type]==T_b)
                            continue;
                        #endif
                        #ifdef TREND
                        if (Property[j][P_type]==T_A)
                            continue;
                        #endif
                        #ifdef BW_BALL
                        if (Property[j][P_type]==T_theta_R || Property[j][P_type]==T_phi_R ||Property[j][P_type]==T_kappa)
                            continue;
                        #endif                        
                        if (Property[j][P_type] == T_Es)
                            fprintf(fmod, "%15.11f ",  E_last);
#ifdef MY_L
                        else if (Property[j][P_type] == T_L)
                            fprintf(fmod, "%15.11f ",  48*PI/L_last);
#endif
                        else
                            fprintf(fmod, "%15.11f ",  params[j]);
                    }
                    fprintf(fmod, "\n");
          #endif
        #endif
        fclose (fmod);
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
            // V is converted to the first filter
            fprintf(fp, "%13.7f %13.6e %13.6e w\n", hMJD0+hData[i].MJD, hData[i].V - h_delta_V[hData[i].Filter] + h_delta_V[0], 1/sqrt(hData[i].w));
        fclose(fp);
        
        #ifdef PROFILES
        char buf[80];
        for (int iparam=0; iparam<N_PARAMS; iparam++)
        {
            snprintf(buf, sizeof(buf), "lines_%i.dat", iparam);
            fp = fopen(buf, "w");
            for (i=0; i<C_POINTS*BSIZE; i++)
            {
                CHI_FLOAT xx = h_chi2_lines[iparam][i];
                if (isnan(xx))
                    xx=1e30;
                fprintf(fp, "%16.9e %16.9e\n", h_param_lines[iparam][i], xx);
            }
            fclose(fp);
        }
        #endif
        #endif  // NOPRINT
    }        
    
    
    
    return 0;  
}
