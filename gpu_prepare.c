#ifdef GPU
#include "asteroid.h"
int gpu_prepare(int N_data, int N_filters, int N_threads)
{

    ERR(cudaMalloc(&d_f, N_BLOCKS * sizeof(CHI_FLOAT)));
    ERR(cudaMalloc(&d_params, N_BLOCKS * sizeof(struct parameters_struct)));

    ERR(cudaMallocHost(&h_f, N_BLOCKS * sizeof(CHI_FLOAT)));
    ERR(cudaMallocHost(&h_params, N_BLOCKS * sizeof(struct parameters_struct)));
    
    ERR(cudaMallocHost(&dData, N_data * sizeof(struct obs_data)));
    
    ERR(cudaMemcpy(dData, hData, N_data * sizeof(struct obs_data), cudaMemcpyHostToDevice));

    
    return 0;
}
#endif
