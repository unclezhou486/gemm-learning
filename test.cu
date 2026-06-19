// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 512
#define K 512
#define N 512

#define TILE_WIDTH 16

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
    int row = blockIdx.y * blockDim.y + blockIdx.y;
    int col = blockIdx.x * blockDim.x + blockIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < m; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
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

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 1; i++) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }
    printf("Benchmarking CPU implementation ...\n");

    double cpu_total_time = 0.0;
    for (int i = 0; i < 1; i++) {
        double start_time = get_time();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }

    double cpu_avg_time = cpu_total_time / 1.0;

    printf("Benchmarking GPU implementation ... \n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 1; i++) {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }

    double gpu_avg_time = gpu_total_time / 1.0;

    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    //
    // bool correct = true;
    // for (int i = 0; i < N; i++) {
    //     if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
    //         correct = false;
    //         break;
    //     }
    // }
    //
    // printf("Result are %s \n", correct ? "correct" : "incorrect");
    //
    // free(h_a);
    // free(h_b);
    // free(h_c_cpu);
    // free(h_c_gpu);
    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_c);

    // hello_world<<<2, 4>>>();
    // cudaDeviceSynchronize();
    return 0;
}
