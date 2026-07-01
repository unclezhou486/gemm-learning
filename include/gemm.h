#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32

#define CUDA_CHECK(err)                                                    \
    do {                                                                   \
        cudaError_t e = (err);                                             \
        if (e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error [%s:%d]: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                                 \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

void matmul_cpu(
    float* A,
    float* B,
    float* C,
    int m,
    int k,
    int n
);

__global__ void matmul_gpu_naive(
    float* A,
    float* B,
    float* C,
    int m,
    int k,
    int n
);

__global__ void matmul_gpu_tiled(
    float* A,
    float* B,
    float* C,
    int m,
    int k,
    int n
);

__global__ void matmul_gpu_thread_tiled_32x32(
    float* A, 
    float* B, 
    float* C, 
    int m, 
    int k, 
    int n
);

__global__ void matmul_gpu_thread_tiled_32x8(
    float* A,
    float* B,
    float* C,
    int m,
    int k,
    int n
);

__global__ void matmul_gpu_thread_tiled_32x16(
    float* A,
    float* B,
    float* C,
    int m,
    int k,
    int n
);
