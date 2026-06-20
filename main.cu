#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gemm.h"

#define M 1024
#define K 1024
#define N 1024

#define EPS 1e-4


void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

bool verify(float* ref,float* out,int size) {
    for(int i=0;i<size;++i){
        if(fabs(ref[i]-out[i])>EPS){
            return false;
        }
    }    
    return true;
}

int main() {

    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // malloc
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);


    // init
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // cudaMalloc
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // warmup

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        // matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_gpu_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        matmul_gpu_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    //benchmark naive
    printf("Benchmarking GPU Naive implementation ...\n");

    double naive_total_time = 0.0;
    int run_times = 20;
    for (int i = 0; i < run_times; i++) {
        double start_time = get_time();
        matmul_gpu_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        // matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        naive_total_time += end_time - start_time;
    }

    double naive_avg_time = naive_total_time / run_times;

    printf("Benchmarking GPU Tiled implementation ... \n");
    double tiled_total_time = 0.0;
    for (int i = 0; i < run_times; i++) {
        double start_time = get_time();
        matmul_gpu_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        tiled_total_time += end_time - start_time;
    }

    double tiled_avg_time = tiled_total_time / run_times;

    printf("GPU Naive average time: %f milliseconds\n", naive_avg_time * 1000);
    printf("GPU Tiled average time: %f milliseconds\n", tiled_avg_time * 1000);
    printf("Speedup: %fx\n", naive_avg_time / tiled_avg_time);

    // verfiy
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    bool correct = verify(h_C_cpu,h_C_gpu,M*N);
    
    printf("Result are %s \n", (correct ? "correct" : "incorrect"));

    // free
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
