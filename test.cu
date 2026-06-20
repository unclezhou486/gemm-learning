// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 1024
#define K 1024
#define N 1024

#define TILE_WIDTH 32

#define EPS 1e-4

void matmul_cpu(float* A, float* B, float* C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__global__ void matmul_gpu(float* A, float* B, float* C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

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

int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C_cpu = (float*)malloc(size_C);
    h_C_gpu = (float*)malloc(size_C);

    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // printf("Performing warm-up runs...\n");
    // for (int i = 0; i < 3; i++) {
    //     matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    //     matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    //     cudaDeviceSynchronize();
    // }
    printf("Benchmarking GPU Naive implementation ...\n");

    double cpu_total_time = 0.0;
    int run_times = 20;
    for (int i = 0; i < run_times; i++) {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        // matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }

    double cpu_avg_time = cpu_total_time / run_times;

    printf("Benchmarking GPU Tiled implementation ... \n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < run_times; i++) {
        double start_time = get_time();
        matmul_gpu_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }

    double gpu_avg_time = gpu_total_time / run_times;

    printf("GPU Naive average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU Tiled average time: %f milliseconds\n", gpu_avg_time * 1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < N * M; i++) {
        if (fabs(h_C_cpu[i] - h_C_gpu[i]) > EPS) {
            // printf("%f %f\n",h_C_cpu[i],h_C_gpu[i]);
            correct = false;
            break;
        }
    }

    printf("Result are %s \n", (correct ? "correct" : "incorrect"));

    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
