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
    // Observational data:
    int N_data; // Number of data points
    int N_filters; // Number of filters used in the data
    
    int j;
    double n_phi, n_x, n_y, n_z;
    
    
    
    // Reading input paameters files
    //    read_input_params();
    
    // Reading all input data files, allocating and initializing observational data arrays   
    read_data("obs.dat", &N_data, &N_filters);
    
    // CPU based chi^2:
    chi2(N_data, N_filters, &chi2tot);
    
    
    return 0;  
}
