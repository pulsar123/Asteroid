#include "asteroid.h"
int gpu_prepare(int N_data, int N_filters, int N_threads, int Nplot)
{

    ERR(cudaMalloc(&d_f, N_BLOCKS * sizeof(CHI_FLOAT)));
//    ERR(cudaMalloc(&d_params, N_BLOCKS * N_PARAMS * sizeof(double)));

    ERR(cudaMallocHost(&h_f, N_BLOCKS * sizeof(CHI_FLOAT)));
//    ERR(cudaMallocHost(&h_params, N_BLOCKS * N_PARAMS * sizeof(double)));
    
    ERR(cudaMallocHost(&dData, N_data * sizeof(struct obs_data)));    
    ERR(cudaMemcpy(dData, hData, N_data * sizeof(struct obs_data), cudaMemcpyHostToDevice));

    if (Nplot > 0)
    {
        ERR(cudaMallocHost(&dPlot, Nplot * sizeof(struct obs_data)));    
        ERR(cudaMemcpy(dPlot, hPlot, Nplot * sizeof(struct obs_data), cudaMemcpyHostToDevice));

        ERR(cudaMalloc(&d_dlsq2, N_data * sizeof(double)));    
        ERR(cudaMallocHost(&h_dlsq2, N_data * sizeof(double)));    
    }
    
#ifdef SEGMENT
//    ERR(cudaMalloc(&d_start_seg, N_SEG * sizeof(int)));
    ERR(cudaMemcpyToSymbol(d_start_seg, h_start_seg, N_SEG * sizeof(int), 0, cudaMemcpyHostToDevice));
    ERR(cudaMemcpyToSymbol(d_plot_start_seg, h_plot_start_seg, N_SEG * sizeof(int), 0, cudaMemcpyHostToDevice));
#endif    

    return 0;
}
