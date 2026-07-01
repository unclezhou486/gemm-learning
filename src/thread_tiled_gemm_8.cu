#include "gemm.h"

__global__ void matmul_gpu_thread_tiled_32x8(float* A, float* B, float* C, int m, int k, int n) {
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int BK = 8;

    // 加上 __align__(16) 确保共享内存 16 字节对齐
    __align__(16) __shared__ float s_A[TILE_WIDTH][BK];
    __align__(16) __shared__ float s_B[BK][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    float r_C[TM][TN] = {0.0f};
    float r_A[TM];
    float r_B[TN];

    A += blockIdx.y * TILE_WIDTH * k;
    B += blockIdx.x * TILE_WIDTH;
    // 【修改点】移除对 C 的指针偏移，避免后面重复计算块偏移

    for (int p = 0; p < (k + BK - 1) / BK; ++p) {
        for (int offset = 0; offset < 4; ++offset) {
            int local_tid = tid + offset * 64;
            int load_row_A = local_tid / BK;
            int load_col_A = local_tid % BK;
            int g_row_A = blockIdx.y * TILE_WIDTH + load_row_A;
            int g_col_A = p * BK + load_col_A;
            if (g_row_A < m && g_col_A < k) {
                s_A[load_row_A][load_col_A] = A[load_row_A * k + p * BK + load_col_A];
            } else {
                s_A[load_row_A][load_col_A] = 0.0f;
            }
        }

        for (int offset = 0; offset < 4; ++offset) {
            int local_tid = tid + offset * 64;
            int load_row_B = local_tid / TILE_WIDTH;
            int load_col_B = local_tid % TILE_WIDTH;
            int g_row_B = p * BK + load_row_B;
            int g_col_B = blockIdx.x * TILE_WIDTH + load_col_B;
            if (g_row_B < k && g_col_B < n) {
                s_B[load_row_B][load_col_B] = B[(p * BK + load_row_B) * n + load_col_B];
            } else {
                s_B[load_row_B][load_col_B] = 0.0f;
            }
        }

        __syncthreads();

        for (int i = 0; i < BK; ++i) {
            for (int r = 0; r < TM; ++r) {
                r_A[r] = s_A[ty * TM + r][i];
            }
            for (int c = 0; c < TN; ++c) {
                r_B[c] = s_B[i][tx * TN + c];
            }
            for (int r = 0; r < TM; ++r) {
                for (int c = 0; c < TN; ++c) {
                    r_C[r][c] += r_A[r] * r_B[c];
                }
            }
        }
        __syncthreads();
    }

    // 这里直接写回未偏移的原始 C 指针中
    for (int r = 0; r < TM; ++r) {
        int g_row_C = blockIdx.y * TILE_WIDTH + ty * TM + r;
        for (int c = 0; c < TN; ++c) {
            int g_col_C = blockIdx.x * TILE_WIDTH + tx * TN + c;
            if (g_row_C < m && g_col_C < n) {
                C[g_row_C * n + g_col_C] = r_C[r][c];
            }
        }
    }
}