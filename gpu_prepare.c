#ifdef GPU
#include "asteroid.h"
int gpu_prepare(int N_data, int N_filters, int N_threads)
{

#ifdef SIMPLEX
    ERR(cudaMalloc(&d_f, N_threads * sizeof(float)));
    ERR(cudaMalloc(&d_params, N_threads * sizeof(struct parameters_struct)));

    ERR(cudaMallocHost(&h_f, N_threads * sizeof(float)));
    ERR(cudaMallocHost(&h_params, N_threads * sizeof(struct parameters_struct)));
#else
    ERR(cudaMalloc(&d_chi2_min, N_BLOCKS * sizeof(float)));
    ERR(cudaMalloc(&d_iloc_min, N_BLOCKS * sizeof(long int)));

    ERR(cudaMallocHost(&h_chi2_min, N_BLOCKS * sizeof(float)));
    ERR(cudaMallocHost(&h_iloc_min, N_BLOCKS * sizeof(long int)));    
#endif
    
    ERR(cudaMallocHost(&dData, N_data * sizeof(struct obs_data)));
    
    ERR(cudaMemcpy(dData, hData, N_data * sizeof(struct obs_data), cudaMemcpyHostToDevice));

    
    return 0;
}
#endif
