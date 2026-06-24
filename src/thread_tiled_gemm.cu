__global__ void matmul_gpu_thread_tiled_32x32(float* A, float* B, float* C, int m, int k, int n) {
    constexpr int TM = 4;        // 每个线程计算的行块
    constexpr int TN = 4;        // 每个线程计算的列块
    constexpr int TILE_WIDTH = 32;  // 正方形分块
    
    // 正方形共享内存
    __align__(16) __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __align__(16) __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 线程分块：每个线程计算4×4
    float r_C[TM][TN] = {0.0f};
    float r_A[TM];
    float r_B[TN];
    
    // 外层循环：遍历K维度
    for (int p = 0; p < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
        
        // 协作加载A矩阵到共享内存
        // block有8×8=64个线程，需要加载32×32=1024个元素
        // 每个线程加载 1024/64 = 16个元素
        
        // 加载A：需要16次展开（或者用循环）
        for (int offset = 0; offset < 16; ++offset) {
            int local_tid = ty * blockDim.x + tx + offset * 64;
            int load_row = local_tid / TILE_WIDTH;  // 0-31
            int load_col = local_tid % TILE_WIDTH;  // 0-31
            
            int g_row = blockIdx.y * TILE_WIDTH + load_row;
            int g_col = p * TILE_WIDTH + load_col;
            
            if (g_row < m && g_col < k) {
                s_A[load_row][load_col] = A[g_row * k + g_col];
            } else {
                s_A[load_row][load_col] = 0.0f;
            }
        }
        
        // 加载B：同样的16次展开
        for (int offset = 0; offset < 16; ++offset) {
            int local_tid = ty * blockDim.x + tx + offset * 64;
            int load_row = local_tid / TILE_WIDTH;  // 0-31
            int load_col = local_tid % TILE_WIDTH;  // 0-31
            
            int g_row = p * TILE_WIDTH + load_row;
            int g_col = blockIdx.x * TILE_WIDTH + load_col;
            
            if (g_row < k && g_col < n) {
                s_B[load_row][load_col] = B[g_row * n + g_col];
            } else {
                s_B[load_row][load_col] = 0.0f;
            }
        }
        
        __syncthreads();
        
        
        for (int i = 0; i < TILE_WIDTH; ++i) {
            // 加载A的4行
            for (int r = 0; r < TM; ++r) {
                r_A[r] = s_A[ty * TM + r][i];
            }
            // 加载B的4列
            for (int c = 0; c < TN; ++c) {
                r_B[c] = s_B[i][tx * TN + c];
            }
            // 4×4外积
            for (int r = 0; r < TM; ++r) {
                for (int c = 0; c < TN; ++c) {
                    r_C[r][c] += r_A[r] * r_B[c];
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回结果
    for (int r = 0; r < TM; ++r) {
        int g_row = blockIdx.y * TILE_WIDTH + ty * TM + r;
        for (int c = 0; c < TN; ++c) {
            int g_col = blockIdx.x * TILE_WIDTH + tx * TN + c;
            if (g_row < m && g_col < n) {
                C[g_row * n + g_col] = r_C[r][c];
            }
        }
    }
}