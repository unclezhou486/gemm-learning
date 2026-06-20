#pragma once
#include <cuda_runtime.h>

#define TILE_WIDTH 32

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