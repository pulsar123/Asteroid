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
    double chi2tot;
    float h_chi2_min=1e30;
    // Observational data:
    int N_data; // Number of data points
    int N_filters; // Number of filters used in the data        
    
    // Reading input paameters files
    //    read_input_params();
    
    // Reading all input data files, allocating and initializing observational data arrays   
    read_data("obs.dat", &N_data, &N_filters);

#ifdef GPU    
    gpu_prepare(N_data, N_filters);
    cudaMemcpyToSymbol(d_chi2_min, &h_chi2_min, sizeof(float), 0, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    chi2_gpu<<<1,1>>>(dData, N_data, N_filters);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("The elapsed time in gpu was %.2f ms\n", elapsed);

    cudaMemcpyFromSymbol(&h_chi2_min, d_chi2_min, sizeof(float), 0, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("GPU ch2_min=%f\n", h_chi2_min);
    
#endif    
    
    // CPU based chi^2:
    chi2(N_data, N_filters, &chi2tot);    
           
    return 0;  
}
