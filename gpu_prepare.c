#ifdef GPU
#include "asteroid.h"
int gpu_prepare(int N_data, int N_filters)
{

    ERR(cudaMalloc(&d_chi2_min, N_BLOCKS * sizeof(double)));
    ERR(cudaMalloc(&d_iloc_min, N_BLOCKS * sizeof(long int)));

    ERR(cudaMallocHost(&h_chi2_min, N_BLOCKS * sizeof(double)));
    ERR(cudaMallocHost(&h_iloc_min, N_BLOCKS * sizeof(long int)));
    
    ERR(cudaMallocHost(&dData, N_data * sizeof(struct obs_data)));
    
    ERR(cudaMemcpy(dData, hData, N_data * sizeof(struct obs_data), cudaMemcpyHostToDevice));
    
    
    return 0;
}
#endif
