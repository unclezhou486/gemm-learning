# gemm-learning

学习 CUDA GEMM 优化的个人仓库，GPU 是 NVIDIA GeForce RTX 4060。

## 硬件环境

GPU 的关键限制决定了所有优化策略的上限，记录下来方便回头查：

- 型号: NVIDIA GeForce RTX 4060
- Compute Capability: 8.9 (sm_86)
- SMs: 24
- Global Memory: 7.63 GB
- L2 Cache: 24 MB
- Max Threads per Block: 1024
- Max Threads per SM: 1536
- Max Warps per SM: 48
- Shared Memory per Block: 48 KB
- Shared Memory per SM: 100 KB
- Registers per Block: 65536
- Registers per SM: 65536
- Warp Size: 32

## 目录结构

```
.
├── main.cu              # 评测入口，跑所有 kernel 并打印对比表
├── profile.cu            # Nsight Compute 用，每次只 launch 一遍 kernel
├── Makefile
├── include/
│   └── gemm.h            # kernel 声明 + CUDA_CHECK 宏 + TILE_WIDTH 常量
├── src/
│   ├── naive_gemm.cu     # 朴素 global memory 实现
│   ├── tiled_gemm.cu     # shared memory 分块
│   ├── thread_tiled_gemm.cu  # shared memory + 寄存器分块
│   └── cpu_gemm.cu       # CPU 参考实现（仅在矩阵较小时使用）
├── information.cu        # 打印 GPU 硬件参数的独立程序
├── information.txt       # information.cu 的输出
└── notes/
    └── LEARNING_ROADMAP.md   # 之前写的完整学习路线图
```

## 构建和运行

```bash
# 编译并跑 benchmark
make && make run

# Nsight Compute 分析单个 kernel
make profile-naive    # 分析 naive kernel
make profile-tiled    # 分析 tiled kernel

# 查看 GPU 信息
nvcc -o info information.cu && ./info
```

## 优化历程

测试矩阵固定为 M=N=K=4096，数据随机初始化，用 FP32 计算。

### 0. Naive：全局内存直读

每个线程直接读 global memory，算一个输出元素。没有任何显式的片上数据复用。

- blockDim: (32, 32), 1024 threads
- gridDim: (128, 128)

这个版本完全是访存瓶颈。4096 的矩阵对于 L2 (24MB) 来说不算大，cache 命中率比预期高，所以并不是慢到不能看。

### 1. Tiled：Shared Memory 分块

把 A 和 B 切成 32x32 的 tile 放进共享内存，同一个 block 内所有线程复用 tile 数据。每个 block 负责 C 的一个 32x32 子块。

- TILE_WIDTH: 32
- blockDim: (32, 32), 1024 threads
- shared memory: s_A[32][32] + s_B[32][32] = 8 KB

加速比有限，主要是两个原因：一是 L2 cache 已经把很多全局内存访问吸收了，shared memory 的边际收益打了折扣；二是 1024 线程 per block 导致每个 SM 只能跑 1 个 block，occupancy 只有 66%。

### 2. Thread Tiled：Shared Memory + 寄存器分块

在 shared memory tiling 的基础上，让每个线程多算几个元素，用寄存器暂存累加结果。具体来说，每个线程算 4x4 个子块，对应的 blockDim 降到 (8,8)=64 线程，但每个 block 仍然负责 32x32 的输出（因为 8x4=32）。

核心改动：

1. 每个线程维护 `r_C[4][4]` 寄存器数组做累加
2. 内层循环里先加载 `r_A[4]` 和 `r_B[4]`，再做 4x4 外积
3. shared memory 的每次加载被 4x4=16 次乘加运算分摊
4. blockDim 降到 64 线程，每 SM 能塞多个 block，occupancy 提升

这一步是到目前为止收益最大的单次改动，加速比相比 Naive 达到 4x 以上。

## Benchmark

矩阵 4096x4096，run_times=10，warmup=2 次。Naive 结果作为参考基准，所有 kernel 和它比对验证。

```text
Kernel Name          | Avg Time (ms)  | Speedup      | GFLOPS       | Verify
-------------------------------------------------------------------------------
Naive                |      168.9502  |       1.00x  |      813.49  | PASS
Tiled                |      127.1272  |       1.33x  |     1081.11  | PASS
Thread Tiled 32x32   |       37.5631  |       4.50x  |     3658.88  | PASS
```


## 待做

- **Vectorized Load (float4)**: 把 global -> shared 的加载从 float 改成 float4，一次搬 16 字节，减少 load 指令条数。需要在 coalescing 和 bounds checking 上做取舍，之前一个版本反而比标量版本慢，需要重新设计寻址部分。
- **Double Buffering**: 用两份 shared memory 交替加载和计算，目的是把 global 加载的延迟藏在计算里，减少 `__syncthreads` 带来的 stall。当前只是纯同步模式，所有线程都要等 tile 加载完才开始算。sm_86 支持 `cp.async`，可以直接做异步拷贝。
- **Tensor Core / Mixed Precision**: 切到 FP16/BF16 用 Tensor Core 算，性能可以再上一个数量级。RTX 4060 的 sm_86 有专门的 tensor core 指令，应该试一下 WMMA 或者直接写 MMA PTX。

## 参考

- [PM Rao: CUDA Matrix Multiplication](https://siboehm.com/articles/22/CUDA-MMM) - 每一步都有代码的可视化教程
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) - 工业级 GEMM 模板库
