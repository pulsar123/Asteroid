#ifdef GPU
#include "asteroid.h"
int gpu_prepare(int N_data, int N_filters, int N_threads, int Nplot)
{

    ERR(cudaMalloc(&d_f, N_BLOCKS * sizeof(CHI_FLOAT)));
    ERR(cudaMalloc(&d_params, N_BLOCKS * sizeof(struct parameters_struct)));
    ERR(cudaMalloc(&d_steps, N_BLOCKS * sizeof(int)));

    ERR(cudaMallocHost(&h_f, N_BLOCKS * sizeof(CHI_FLOAT)));
    ERR(cudaMallocHost(&h_params, N_BLOCKS * sizeof(struct parameters_struct)));
    ERR(cudaMallocHost(&h_steps, N_BLOCKS * sizeof(int)));
    
    ERR(cudaMallocHost(&dData, N_data * sizeof(struct obs_data)));    
    ERR(cudaMemcpy(dData, hData, N_data * sizeof(struct obs_data), cudaMemcpyHostToDevice));

    if (Nplot > 0)
    {
        ERR(cudaMallocHost(&dPlot, Nplot * sizeof(struct obs_data)));    
        ERR(cudaMemcpy(dPlot, hPlot, Nplot * sizeof(struct obs_data), cudaMemcpyHostToDevice));

        ERR(cudaMalloc(&d_dlsq2, N_data * sizeof(double)));    
        ERR(cudaMallocHost(&h_dlsq2, N_data * sizeof(double)));    
    }
    
    return 0;
}
#endif
