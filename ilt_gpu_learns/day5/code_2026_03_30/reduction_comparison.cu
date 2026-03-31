/*
 * 归约算法性能对比
 * 
 * 本程序对比三种GPU归约算法的性能
 * 展示不同优化策略的效果
 * 
 * 方法：
 * 1. 朴素归约（原子操作）
 * 2. 共享内存归约（树形归约）
 * 3. 循环展开归约（Warp级优化）
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (16 * 1024 * 1024)  // 16M元素

// 错误检查宏
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ============ CPU归约（基准） ============
float reduceCPU(float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// ============ 方法1: 朴素归约（原子操作） ============
/*
 * 使用原子操作实现归约
 * 
 * 优点：代码简单
 * 缺点：原子操作串行化，性能差
 * 
 * 性能分析：
 * - 所有线程竞争同一个内存地址
 * - 原子操作必须串行执行
 * - 并发度极低
 */
__global__ void reduceNaive(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 所有线程都竞争output地址
        // 串行化执行，性能最差
        atomicAdd(output, input[idx]);
    }
}

// ============ 方法2: 共享内存归约 ============
/*
 * 使用共享内存进行块内归约
 * 
 * 优化原理：
 * 1. 每个线程块独立归约
 * 2. 使用共享内存减少全局内存访问
 * 3. 树形结构减少同步次数
 * 
 * 性能提升：比朴素方法快10-20倍
 */
__global__ void reduceShared(float* input, float* partialSums, int n) {
    __shared__ float sdata[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 加载数据到共享内存
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // 树形归约
    // 第1轮：256 -> 128
    // 第2轮：128 -> 64
    // ...
    // 最后：2 -> 1
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 只有线程0写入结果
    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

// ============ 方法3: 循环展开归约 ============
/*
 * 在共享内存归约基础上，展开最后几轮循环
 * 
 * 优化原理：
 * 1. 展开循环减少指令开销
 * 2. 最后几轮循环时，线程束内的线程数量较少
 * 3. 使用Warp级指令，避免同步开销
 * 
 * 性能提升：比方法2快1.2-1.5倍
 */
__global__ void reduceUnrolled(float* input, float* partialSums, int n) {
    __shared__ float sdata[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 加载数据
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // 展开循环
    // 当s >= 64时，使用常规循环
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 最后的Warp归约（展开）
    // Warp大小为32，此时只有前32个线程活跃
    // 不需要__syncthreads()，因为Warp是同步执行的
    if (tid < 32) {
        // 手动展开最后5轮（32 -> 16 -> 8 -> 4 -> 2 -> 1）
        sdata[tid] += sdata[tid + 32];  // 32 -> 16
        sdata[tid] += sdata[tid + 16];  // 16 -> 8
        sdata[tid] += sdata[tid + 8];   // 8 -> 4
        sdata[tid] += sdata[tid + 4];   // 4 -> 2
        sdata[tid] += sdata[tid + 2];   // 2 -> 1
        sdata[tid] += sdata[tid + 1];   // 最后一个元素
    }
    
    // 写入结果
    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

// ============ 方法4: Warp Shuffle归约（高级） ============
/*
 * 使用Warp Shuffle指令进行归约
 * 
 * 优化原理：
 * 1. 使用__shfl_down_sync直接在寄存器间交换数据
 * 2. 避免共享内存访问
 * 3. 更快的归约速度
 * 
 * 性能提升：比方法3快1.1-1.2倍
 * 
 * 注意：需要计算能力 >= 3.0
 */
__global__ void reduceWarpShuffle(float* input, float* partialSums, int n) {
    __shared__ float sdata[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 加载数据
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp级归约
    // 使用shuffle指令在Warp内归约
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // 每个Warp的第一个线程写入共享内存
    if (tid % 32 == 0) {
        sdata[tid / 32] = val;
    }
    __syncthreads();
    
    // 最后一个Warp归约共享内存中的结果
    if (tid < 32) {
        val = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (tid == 0) {
            partialSums[blockIdx.x] = val;
        }
    }
}

// ============ 辅助函数 ============
void initData(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 1.0f;  // 使用1.0，方便验证结果
    }
}

// ============ 主函数 ============
int main() {
    printf("=== 归约算法性能对比 ===\n\n");
    printf("向量大小: %d 元素 (%.2f MB)\n\n", N, N * sizeof(float) / 1024.0 / 1024.0);
    
    size_t size = N * sizeof(float);
    
    // ========== 分配主机内存 ==========
    float *h_data = (float*)malloc(size);
    
    // ========== 初始化数据 ==========
    initData(h_data, N);
    
    // ========== CPU归约 ==========
    printf("计算CPU基准结果...\n");
    clock_t cpu_start = clock();
    float cpu_result = reduceCPU(h_data, N);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU结果: %.2f\n", cpu_result);
    printf("CPU时间: %.2f ms\n\n", cpu_time);
    
    // ========== 分配设备内存 ==========
    float *d_data, *d_partialSums, *d_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_partialSums, sizeof(float) * 65536));  // 足够大
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    // ========== 创建CUDA Event ==========
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // 线程块配置
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // ========== 方法1: 朴素归约 ==========
    printf("方法1 - 朴素归约（原子操作）:\n");
    CHECK_CUDA_ERROR(cudaMemset(d_result, 0, sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    reduceNaive<<<numBlocks, blockSize>>>(d_data, d_result, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float naive_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&naive_time, start, stop));
    
    float result;
    CHECK_CUDA_ERROR(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    float bandwidth_naive = (size / naive_time) / 1e6;  // GB/s
    printf("  时间: %.2f ms\n", naive_time);
    printf("  带宽: %.2f GB/s\n", bandwidth_naive);
    printf("  结果: %.2f (%s)\n\n", result, fabs(cpu_result - result) < 1.0f ? "正确" : "错误");
    
    // ========== 方法2: 共享内存归约 ==========
    printf("方法2 - 共享内存归约:\n");
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    reduceShared<<<numBlocks, blockSize>>>(d_data, d_partialSums, N);
    
    // 第二阶段：归约部分和
    float *h_partialSums = (float*)malloc(numBlocks * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    float shared_result = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        shared_result += h_partialSums[i];
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float shared_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&shared_time, start, stop));
    
    float bandwidth_shared = (size / shared_time) / 1e6;
    printf("  时间: %.2f ms\n", shared_time);
    printf("  带宽: %.2f GB/s\n", bandwidth_shared);
    printf("  加速比: %.2fx\n", naive_time / shared_time);
    printf("  结果: %.2f (%s)\n\n", shared_result, fabs(cpu_result - shared_result) < 1.0f ? "正确" : "错误");
    
    // ========== 方法3: 循环展开归约 ==========
    printf("方法3 - 循环展开归约:\n");
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    reduceUnrolled<<<numBlocks, blockSize>>>(d_data, d_partialSums, N);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    float unrolled_result = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        unrolled_result += h_partialSums[i];
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float unrolled_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&unrolled_time, start, stop));
    
    float bandwidth_unrolled = (size / unrolled_time) / 1e6;
    printf("  时间: %.2f ms\n", unrolled_time);
    printf("  带宽: %.2f GB/s\n", bandwidth_unrolled);
    printf("  加速比: %.2fx (vs 方法1)\n", naive_time / unrolled_time);
    printf("  加速比: %.2fx (vs 方法2)\n", shared_time / unrolled_time);
    printf("  结果: %.2f (%s)\n\n", unrolled_result, fabs(cpu_result - unrolled_result) < 1.0f ? "正确" : "错误");
    
    // ========== 方法4: Warp Shuffle归约 ==========
    printf("方法4 - Warp Shuffle归约:\n");
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    reduceWarpShuffle<<<numBlocks, blockSize>>>(d_data, d_partialSums, N);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    float shuffle_result = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        shuffle_result += h_partialSums[i];
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float shuffle_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&shuffle_time, start, stop));
    
    float bandwidth_shuffle = (size / shuffle_time) / 1e6;
    printf("  时间: %.2f ms\n", shuffle_time);
    printf("  带宽: %.2f GB/s\n", bandwidth_shuffle);
    printf("  加速比: %.2fx (vs 方法1)\n", naive_time / shuffle_time);
    printf("  加速比: %.2fx (vs 方法2)\n", shared_time / shuffle_time);
    printf("  加速比: %.2fx (vs 方法3)\n", unrolled_time / shuffle_time);
    printf("  结果: %.2f (%s)\n\n", shuffle_result, fabs(cpu_result - shuffle_result) < 1.0f ? "正确" : "错误");
    
    // ========== 性能总结 ==========
    printf("=== 性能总结 ===\n");
    printf("方法1 (原子操作):     %8.2f ms  带宽: %6.2f GB/s\n", naive_time, bandwidth_naive);
    printf("方法2 (共享内存):     %8.2f ms  带宽: %6.2f GB/s  加速: %5.2fx\n", 
           shared_time, bandwidth_shared, naive_time / shared_time);
    printf("方法3 (循环展开):     %8.2f ms  带宽: %6.2f GB/s  加速: %5.2fx\n", 
           unrolled_time, bandwidth_unrolled, naive_time / unrolled_time);
    printf("方法4 (Warp Shuffle): %8.2f ms  带宽: %6.2f GB/s  加速: %5.2fx\n", 
           shuffle_time, bandwidth_shuffle, naive_time / shuffle_time);
    
    printf("\n优化技巧总结:\n");
    printf("1. 原子操作 -> 共享内存: 避免串行化，提升 %.2fx\n", naive_time / shared_time);
    printf("2. 常规循环 -> 循环展开: 减少指令开销，提升 %.2fx\n", shared_time / unrolled_time);
    printf("3. 共享内存 -> Warp Shuffle: 避免共享内存访问，提升 %.2fx\n", 
           unrolled_time / shuffle_time);
    
    // ========== 清理资源 ==========
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaFree(d_partialSums));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    
    free(h_data);
    free(h_partialSums);
    
    printf("\n程序执行完成！\n");
    return 0;
}

/*
 * ========== 编译和运行 ==========
 * 
 * 编译:
 *   nvcc -O3 -arch=sm_80 reduction_comparison.cu -o reduction
 * 
 * 运行:
 *   ./reduction
 * 
 * 性能分析:
 *   nvprof ./reduction
 * 
 * ========== 预期输出 ==========
 * 
 * 方法1: ~2-3 ms, 带宽 ~20-30 GB/s
 * 方法2: ~1.2-1.5 ms, 带宽 ~50-60 GB/s, 加速 ~2x
 * 方法3: ~0.9-1.2 ms, 带宽 ~60-70 GB/s, 加速 ~2.5x
 * 方法4: ~0.7-0.9 ms, 带宽 ~70-90 GB/s, 加速 ~3x
 */
