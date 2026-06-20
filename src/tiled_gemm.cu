#include "gemm.h"

__global__ void matmul_gpu_tiled(float* A, float* B, float* C, int m, int k,
                                 int n) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;
    
    float sum = 0.0f;

    for(int p=0;p<(k+TILE_WIDTH-1)/TILE_WIDTH;++p) {
        if(row<m&&(p*TILE_WIDTH+tx)<k)
            s_A[ty][tx] = A[row*k+p*TILE_WIDTH+tx];
        else 
            s_A[ty][tx]=0.0f;
        if(col<n&&(p*TILE_WIDTH+ty)<k)
            s_B[ty][tx] = B[(p*TILE_WIDTH+ty)*n+col];
        else
            s_B[ty][tx]=0.0f;
        __syncthreads();
        for(int i=0;i<TILE_WIDTH;++i) {
            sum+=s_A[ty][i] * s_B[i][tx];
        }
        __syncthreads();
    }
    if(row<m && col < n){
        C[row * n + col] = sum;
    }
}