#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gemm.h"

#define M 4096
#define K 4096
#define N 4096

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

    // malloc
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_ref = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    // init
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // cudaMalloc
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // warmup

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmul_gpu_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        matmul_gpu_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // benchmark naive (also serves as reference for tiled verification)
    printf("Benchmarking GPU Naive implementation ...\n");

    cudaEvent_t start_1, stop_1;
    CUDA_CHECK(cudaEventCreate(&start_1));
    CUDA_CHECK(cudaEventCreate(&stop_1));

    int run_times = 1;

    CUDA_CHECK(cudaEventRecord(start_1));
    for (int i = 0; i < run_times; i++) {
        matmul_gpu_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop_1));
    CUDA_CHECK(cudaEventSynchronize(stop_1));

    float naive_total_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_total_time, start_1, stop_1));

    double naive_avg_time = naive_total_time / run_times;

    // stash naive result as reference (avoids slow CPU matmul on large sizes)
    CUDA_CHECK(cudaMemcpy(h_C_ref, d_C, size_C, cudaMemcpyDeviceToHost));

    // benchmark tiled
    printf("Benchmarking GPU Tiled implementation ...\n");

    cudaEvent_t start_2, stop_2;
    CUDA_CHECK(cudaEventCreate(&start_2));
    CUDA_CHECK(cudaEventCreate(&stop_2));

    CUDA_CHECK(cudaEventRecord(start_2));
    for (int i = 0; i < run_times; i++) {
        matmul_gpu_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop_2));
    CUDA_CHECK(cudaEventSynchronize(stop_2));

    float tiled_total_time;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_total_time, start_2, stop_2));
    double tiled_avg_time = tiled_total_time / run_times;

    printf("GPU Naive average time: %f milliseconds\n", naive_avg_time);
    printf("GPU Tiled average time: %f milliseconds\n", tiled_avg_time);
    printf("Speedup: %fx\n", naive_avg_time / tiled_avg_time);

    double naive_gflops = (2.0 * M * K * N) / (naive_avg_time / 1000.0) / 1e9;
    double tiled_gflops = (2.0 * M * K * N) / (tiled_avg_time / 1000.0) / 1e9;
    printf("Naive GFLOPS: %.2f\n", naive_gflops);
    printf("Tiled GFLOPS: %.2f\n", tiled_gflops);

    // verify: compare tiled output against naive reference
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
    bool ok = verify(h_C_ref, h_C_gpu, M * N);
    printf("Tiled result: %s\n", ok ? "PASS" : "FAIL");

    // free
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_gpu);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start_1));
    CUDA_CHECK(cudaEventDestroy(stop_1));
    CUDA_CHECK(cudaEventDestroy(start_2));
    CUDA_CHECK(cudaEventDestroy(stop_2));

    return 0;
}
