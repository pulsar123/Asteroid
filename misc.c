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


int quadratic_interpolation(double MJD, OBS_TYPE *E_x1, OBS_TYPE *E_y1, OBS_TYPE *E_z1, OBS_TYPE *S_x1, OBS_TYPE *S_y1, OBS_TYPE *S_z1)
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


#ifdef NUDGE
int prepare_chi2_params(int * N_data)
{
    struct chi2_struct h_chi2_params;
    char line[MAX_LINE_LENGTH];
    float t_obs, V_obs;
    
    FILE *f1 = fopen("observed.min","r");
    
    int i = -1;
    while (fgets(line, sizeof(line), f1)) 
    {
        // The brightness minima times and magnitudes, in converted coordinates (the ones used to compute chi2)
        sscanf(line, "%f %f", &t_obs, &V_obs);
        const double small=0.05;
        if (t_obs >= hMJD0-small && t_obs <= hData[*N_data-1].MJD+hMJD0+small)
            // Only keeping the minima which are within the observed range
        {
            i++;
            if (i >= NOBS_MAX)
            {
                printf ("Too many lines in observed.min file! (>NOBS_MAX)!\n");
                exit (1);
            }
            h_chi2_params.t_obs[i] = t_obs - hMJD0;
            h_chi2_params.V_obs[i] = V_obs;
        }
    }
    fclose(f1);
    
    // Number of observed minima:
    h_chi2_params.N_obs = i + 1;
    if (h_chi2_params.N_obs < 1)
    {
        printf ("No local minima in the data range!\n");
        exit (1);
    }
    printf ("%d minima in observed.min\n", h_chi2_params.N_obs);
    
    // Copying the observed minima data to GPU:
    ERR(cudaMemcpyToSymbol(d_chi2_params, &h_chi2_params, sizeof(struct chi2_struct), 0, cudaMemcpyHostToDevice));
    
    return 0;
}
#endif


#ifdef MINIMA_TEST
int minima_test(int N_data, int N_filters, int Nplot, double* params, int Types[][N_SEG], CHI_FLOAT delta_V)
/*  Counting deep minima for different theta_M, phi_M (and phi_0?) parameters. To judge how likley disk vs. cigar models are.
 */
{
    
    dim3 NB (N_THETA_M, N_PHI_M);
            
    // Copying the model parameters to the gpu:
    ERR(cudaMemcpyToSymbol(d_params0, params, N_PARAMS*sizeof(double), 0, cudaMemcpyHostToDevice));
    h_N7all = 0;
    ERR(cudaMemcpyToSymbol(d_N7all, &h_N7all, sizeof(int), 0, cudaMemcpyHostToDevice));
        
    // Computing the score matrix:
    chi2_minima<<<NB, N_PHI_0>>>(dData, N_data, N_filters, dPlot, Nplot, delta_V);
        
    // Copying the score matrix to the host:
    ERR(cudaMemcpyFromSymbol(&h_Scores, d_Scores, N_THETA_M*N_PHI_M*sizeof(float), 0, cudaMemcpyDeviceToHost));
    ERR(cudaMemcpyFromSymbol(&h_Prob, d_Prob, N_THETA_M*N_PHI_M*sizeof(float), 0, cudaMemcpyDeviceToHost));
    ERR(cudaMemcpyFromSymbol(&h_N7all, d_N7all, sizeof(int), 0, cudaMemcpyDeviceToHost));
    ERR(cudaDeviceSynchronize());

    FILE* fmap = fopen("minima_map.dat", "w");
    
    double sum = 0.0;
    for (int i=0; i<N_THETA_M; i++)
    {
        for (int j=0; j<N_PHI_M; j++)
        {
            sum = sum + h_Scores[i][j];
            fprintf(fmap, "%f ", h_Prob[i][j]); // Writing the probability map
        }
        fprintf(fmap, "\n");
    }
    fclose(fmap);
    
    printf("Average score: %f\n", sum/(double)N_THETA_M / (double)N_PHI_M);
    printf("Model likelihood: %lf\n", (double)h_N7all / (N_THETA_M*N_PHI_M*N_PHI_0));
    
    
    
    return 0;
}
#endif


#ifdef ANIMATE
int write_PNGs(unsigned char * h_rgb, int i1, int i2)
{
    char filename[256];

    int height = SIZE_PIX;
    int width = SIZE_PIX;
    
    int code = 0;
    FILE *fp = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_bytep row = NULL;
    
    for (int i_snapshot=i1; i_snapshot<i2; i_snapshot++)
    {
        sprintf(filename, "image_%05d.png", i_snapshot);
        
        // Open file for writing (binary mode)
        fp = fopen(filename, "wb");
        if (fp == NULL) {
            fprintf(stderr, "Could not open file %s for writing\n", filename);
            return 1;
        }
        
        // Initialize write structure
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (png_ptr == NULL) {
            fprintf(stderr, "Could not allocate write struct\n");
            return 1;
        }
        
        // Initialize info structure
        info_ptr = png_create_info_struct(png_ptr);
        if (info_ptr == NULL) {
            fprintf(stderr, "Could not allocate info struct\n");
            return 1;
        }
        
        // Setup Exception handling
        if (setjmp(png_jmpbuf(png_ptr))) {
            fprintf(stderr, "Error during png creation\n");
            return 1;
        }
        
        png_init_io(png_ptr, fp);
        
        // Write header (8 bit colour depth)
        png_set_IHDR(png_ptr, info_ptr, width, height,
                     8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
        
        png_write_info(png_ptr, info_ptr);
        
        // Allocate memory for one row (3 bytes per pixel - RGB)
        row = (png_bytep) malloc(3 * width * sizeof(png_byte));
        
        // Write image data
        int x, y;
        long int iflat;
        for (y=0 ; y<height ; y++) 
        {
            for (x=0 ; x<width ; x++) 
            {
                for (int ic=0; ic<3; ic++)
                {
                    // Flattened array index:
                    iflat = ic + 3*(x + (long int)SIZE_PIX*(y + (long int)SIZE_PIX*(i_snapshot-i1)));
                    row[x*3+ic] = (png_byte)(h_rgb[iflat]);
                }
            }
            png_write_row(png_ptr, row);
        }
        
        // End write
        png_write_end(png_ptr, NULL);
        
        if (fp != NULL) fclose(fp);
        if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
        if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        if (row != NULL) free(row);
        
    }  // snapshot for loop
    
    return code;
    
    
}
#endif
