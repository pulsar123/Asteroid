#include "asteroid.h"
int gpu_prepare(int N_data, int N_filters, int N_threads, int Nplot)
{

    ERR(cudaMalloc(&d_f, N_BLOCKS * sizeof(CHI_FLOAT)));
    ERR(cudaMalloc(&d_params, N_BLOCKS * N_PARAMS * sizeof(double)));
    ERR(cudaMalloc(&d_dV, N_BLOCKS * N_FILTERS * sizeof(double)));

    ERR(cudaMallocHost(&h_f, N_BLOCKS * sizeof(CHI_FLOAT)));
    ERR(cudaMallocHost(&h_params, N_BLOCKS * N_PARAMS * sizeof(double)));
    ERR(cudaMallocHost(&h_dV, N_BLOCKS * N_FILTERS * sizeof(double)));
    
    ERR(cudaMallocHost(&dData, N_data * sizeof(struct obs_data)));    
    ERR(cudaMemcpy(dData, hData, N_data * sizeof(struct obs_data), cudaMemcpyHostToDevice));

#ifdef RMSD
    ERR(cudaMalloc(&dpar_min, N_BLOCKS * N_PARAMS * sizeof(float)));
    ERR(cudaMalloc(&dpar_max, N_BLOCKS * N_PARAMS * sizeof(float)));
    ERR(cudaMallocHost(&hpar_min, N_BLOCKS * N_PARAMS * sizeof(float)));
    ERR(cudaMallocHost(&hpar_max, N_BLOCKS * N_PARAMS * sizeof(float)));
#endif    
    
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

#ifdef INTERP
    ERR(cudaMemcpyToSymbol(dE_x0, E_x0, 3*sizeof(double), 0, cudaMemcpyHostToDevice));
    ERR(cudaMemcpyToSymbol(dE_y0, E_y0, 3*sizeof(double), 0, cudaMemcpyHostToDevice));
    ERR(cudaMemcpyToSymbol(dE_z0, E_z0, 3*sizeof(double), 0, cudaMemcpyHostToDevice));
    ERR(cudaMemcpyToSymbol(dS_x0, S_x0, 3*sizeof(double), 0, cudaMemcpyHostToDevice));
    ERR(cudaMemcpyToSymbol(dS_y0, S_y0, 3*sizeof(double), 0, cudaMemcpyHostToDevice));
    ERR(cudaMemcpyToSymbol(dS_z0, S_z0, 3*sizeof(double), 0, cudaMemcpyHostToDevice));
    ERR(cudaMemcpyToSymbol(dMJD0, MJD0, 3*sizeof(double), 0, cudaMemcpyHostToDevice));
#endif    
    
#ifdef ANIMATE
    long int size = 3 * (long int)dNPLOT * (long int)SIZE_PIX * (long int)SIZE_PIX * sizeof(unsigned char);
    printf("d_rgb size: %f GB\n", size/1024.0/1024.0/1024.0);
    ERR(cudaMalloc(&d_rgb, size));
    ERR(cudaMallocHost(&h_rgb, size));
#endif    
    
    return 0;
}
