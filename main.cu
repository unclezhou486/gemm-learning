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

struct KernelInfo {
    const char* name;
    void (&func)(float* ,float* ,float* ,int ,int ,int);
    dim3 block;
    dim3 grid;
};


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

    dim3 block_32_32(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_32_32((N+TILE_WIDTH-1)/TILE_WIDTH,
                    (M+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 block_8_8(8, 8);
    dim3 grid_8_8((N+TILE_WIDTH-1)/TILE_WIDTH,
                    (M+TILE_WIDTH-1)/TILE_WIDTH);

    // 
    KernelInfo kernels[] = {
        {"Naive",                   matmul_gpu_naive,                   block_32_32,    grid_32_32},
        {"Tiled",                   matmul_gpu_tiled,                   block_32_32,    grid_32_32},
        {"Thread Tiled x32 ",     matmul_gpu_thread_tiled_32x32 ,     block_8_8,      grid_8_8},
        {"Thread Tiled x16 ",     matmul_gpu_thread_tiled_32x16 ,     block_8_8,      grid_8_8},
        {"Thread Tiled x8 ",     matmul_gpu_thread_tiled_32x8 ,     block_8_8,      grid_8_8},

    };
    int num_kernels = sizeof(kernels) / sizeof(kernels[0]);
    //1. Naive to ref
    printf("Generating baseline reference result via Naive kernel...\n");
    CUDA_CHECK(cudaMemset(d_C,0,size_C));
    kernels[0].func<<<kernels[0].grid,kernels[0].block>>>(d_A,d_B,d_C,M,K,N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_ref,d_C,size_C,cudaMemcpyDeviceToHost));


    //2. warmup

    printf("Performing warm-up runs for all kernels ...\n");
    
    for(int i=0;i<num_kernels;++i){
        for(int j=0;j<2;j++){
            kernels[i].func<<<kernels[i].grid,kernels[i].block>>>(d_A,d_B,d_C,M,K,N);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    //3.benchmark

    int run_times = 10;

    float results_avg_time[num_kernels] = {0};
    bool results_verify[num_kernels] = {false};


    printf("\nStarting benchmark (run_times = %d)...\n", run_times);
    for (int k = 0; k < num_kernels; k++) {
        printf("Benchmarking %s ...\n", kernels[k].name);
        
        // 每次测试前清零目标显存，防止错误的实现因为继承前一个算子的旧值而通过测试
        CUDA_CHECK(cudaMemset(d_C, 0, size_C));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < run_times; i++) {
            kernels[k].func<<<kernels[k].grid, kernels[k].block>>>(d_A, d_B, d_C, M, K, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float total_time;
        CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));
        results_avg_time[k] = total_time / run_times;

        // 拷贝结果回主机并验证正确性
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));
        results_verify[k] = verify(h_C_ref, h_C_gpu, M * N);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // --- 4. 汇总并打印对比报告 ---
    printf("\n================================= GEMM BENCHMARK REPORT =================================\n");
    printf("%-18s | %-15s | %-10s | %-12s | %-10s\n", "Kernel Name", "Avg Time (ms)", "Speedup", "GFLOPS", "Verify");
    printf("-----------------------------------------------------------------------------------------\n");
    
    double naive_time = results_avg_time[0];
    for (int k = 0; k < num_kernels; k++) {
        double avg_time = results_avg_time[k];
        double speedup = naive_time / avg_time;
        // 矩阵乘法的 FLOP 计算公式为: 2 * M * K * N
        double gflops = (2.0 * M * K * N) / (avg_time / 1000.0) / 1e9;
        printf("%-18s | %15.4f | %9.2fx | %12.2f | %-10s\n", 
               kernels[k].name, 
               avg_time, 
               speedup, 
               gflops, 
               results_verify[k] ? "PASS" : "FAIL");
    }
    printf("=========================================================================================\n");

    // free
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_gpu);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
