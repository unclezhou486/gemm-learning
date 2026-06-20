#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gemm.h"

// #define M 4096
// #define K 4096
// #define N 4096

#ifndef M
#define M 1024
#endif
#ifndef K
#define K 1024
#endif
#ifndef N
#define N 1024
#endif

#define EPS 1e-4

void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

bool verify(float* ref, float* out, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(ref[i] - out[i]) > EPS) {
            return false;
        }
    }
    return true;
}

int main() {
    float *h_A, *h_B, *h_C_ref, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // host malloc
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_ref = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    // init
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // device alloc + copy
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // ---- naive kernel (single launch for ncu) ----
    printf("=== Profiling: matmul_gpu_naive ===\n");
    matmul_gpu_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, size_C, cudaMemcpyDeviceToHost));
    printf("naive launch done, result saved as reference\n");

    // ---- tiled kernel (single launch for ncu) ----
    printf("=== Profiling: matmul_gpu_tiled ===\n");
    matmul_gpu_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
    printf("tiled launch done\n");

    // verify
    bool ok = verify(h_C_ref, h_C_gpu, M * N);
    printf("Tiled vs Naive: %s\n", ok ? "PASS" : "FAIL");

    // cleanup
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_gpu);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
