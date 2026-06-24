#include <cuda_runtime.h>
#include <stdio.h>

void print_gpu_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("=== GPU: %s ===\n", prop.name);
    
    // 计算能力
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    // 内存限制
    printf("\n--- 内存限制 ---\n");
    printf("Global Memory: %.2f GB\n", 
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %zu KB\n", 
           prop.sharedMemPerBlock / 1024);
    printf("Shared Memory per SM: %zu KB\n", 
           prop.sharedMemPerMultiprocessor / 1024);
    
    // 寄存器限制
    printf("\n--- 寄存器限制 ---\n");
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    
    // 线程限制
    printf("\n--- 线程限制 ---\n");
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Block Dimensions: (%d, %d, %d)\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Dimensions: (%d, %d, %d)\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    // SM信息
    printf("\n--- SM 信息 ---\n");
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Warp Size: %d\n", prop.warpSize);
    
    // 其他重要限制
    printf("\n--- 其他限制 ---\n");
    printf("Max Warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);
    printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("Constant Memory: %zu KB\n", prop.totalConstMem / 1024);
}

int main() {
    print_gpu_info();
    return 0;
}