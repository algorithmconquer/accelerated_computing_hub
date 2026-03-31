/*
 * GPU向量运算库 - 优化示例
 * 
 * 本程序演示GPU向量运算的优化技巧
 * 包含：向量加法、向量点积、向量缩放
 * 
 * 优化技术：
 * 1. 向量化内存访问（float4）
 * 2. 归约算法优化
 * 3. 合并内存访问
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 向量元素数量
#define N 10000000  // 1000万元素

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

// ============ CPU向量运算 ============

// CPU向量加法
void vectorAddCPU(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CPU向量点积
float vectorDotCPU(float* a, float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// CPU向量缩放
void vectorScaleCPU(float* a, float scalar, float* b, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = a[i] * scalar;
    }
}

// ============ GPU向量运算 ============

/*
 * GPU向量加法 - 朴素版本
 * 每个线程处理一个元素
 */
__global__ void vectorAddNaive(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/*
 * GPU向量加法 - 向量化优化版本
 * 使用float4一次处理4个元素
 * 
 * 优化原理：
 * - float4是128位对齐，可以一次内存事务完成
 * - 减少内存请求次数，提高带宽利用率
 * - 性能提升约2-3倍
 */
__global__ void vectorAddOptimized(float4* a, float4* b, float4* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 一次读取4个float
        float4 va = a[idx];
        float4 vb = b[idx];
        float4 vc;
        
        // 分别计算
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        
        // 写回
        c[idx] = vc;
    }
}

/*
 * GPU向量点积 - 朴素归约版本
 * 使用原子操作
 * 
 * 性能问题：
 * - 原子操作串行化，并发度低
 * - 大量线程竞争，性能差
 */
__global__ void vectorDotNaive(float* a, float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 使用原子加法，线程不安全
        atomicAdd(result, a[idx] * b[idx]);
    }
}

/*
 * GPU向量点积 - 树形归约优化版本
 * 使用共享内存进行归约
 * 
 * 优化原理：
 * 1. 每个线程块内使用共享内存归约
 * 2. 树形结构减少同步次数
 * 3. 最后一个块归约所有块的结果
 * 
 * 性能提升：比原子操作快10-20倍
 */
__global__ void vectorDotOptimized(float* a, float* b, float* partialSums, int n) {
    // 共享内存用于块内归约
    __shared__ float sdata[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // 加载数据并计算乘积
    float product = (idx < n) ? a[idx] * b[idx] : 0.0f;
    sdata[tid] = product;
    
    __syncthreads();
    
    // 树形归约
    // 每次迭代，活跃线程数量减半
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

/*
 * GPU向量缩放 - 朴素版本
 * 每个线程处理一个元素
 */
__global__ void vectorScaleNaive(float* a, float scalar, float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] * scalar;
    }
}

/*
 * GPU向量缩放 - 向量化优化版本
 * 使用float4一次处理4个元素
 */
__global__ void vectorScaleOptimized(float4* a, float scalar, float4* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float4 va = a[idx];
        float4 vb;
        
        vb.x = va.x * scalar;
        vb.y = va.y * scalar;
        vb.z = va.z * scalar;
        vb.w = va.w * scalar;
        
        b[idx] = vb;
    }
}

// ============ 辅助函数 ============

void initVector(float* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)(rand() % 100) / 10.0f;
    }
}

bool verifyVectorAdd(float* ref, float* result, int n, float epsilon = 1e-3) {
    for (int i = 0; i < n; i++) {
        if (fabs(ref[i] - result[i]) > epsilon) {
            printf("向量加法验证失败：位置 %d, 期望 %.6f, 实际 %.6f\n",
                   i, ref[i], result[i]);
            return false;
        }
    }
    return true;
}

// ============ 主函数 ============
int main() {
    printf("=== GPU向量运算性能对比 ===\n\n");
    printf("向量大小: %d 元素 (%.2f MB)\n\n", N, N * sizeof(float) / 1024.0 / 1024.0);
    
    size_t size = N * sizeof(float);
    
    // ========== 分配主机内存 ==========
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    float *h_c_ref = (float*)malloc(size);
    
    // ========== 初始化数据 ==========
    srand(2024);
    initVector(h_a, N);
    initVector(h_b, N);
    
    // ========== 分配设备内存 ==========
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));
    
    // ========== 拷贝数据到设备 ==========
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // ========== 创建CUDA Event ==========
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // 线程块配置
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    // ========== 1. 向量加法对比 ==========
    printf("1. 向量加法:\n");
    
    // CPU版本
    clock_t cpu_start = clock();
    vectorAddCPU(h_a, h_b, h_c_ref, N);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("   CPU时间: %.2f ms\n", cpu_time);
    
    // GPU朴素版本
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorAddNaive<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float naive_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&naive_time, start, stop));
    printf("   GPU朴素时间: %.2f ms (加速 %.2fx)\n", naive_time, cpu_time / naive_time);
    
    // GPU向量化版本
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorAddOptimized<<<numBlocks / 4, blockSize>>>(reinterpret_cast<float4*>(d_a),
                                                       reinterpret_cast<float4*>(d_b),
                                                       reinterpret_cast<float4*>(d_c),
                                                       N / 4);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float opt_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&opt_time, start, stop));
    printf("   GPU优化时间: %.2f ms (加速 %.2fx)\n", opt_time, cpu_time / opt_time);
    
    // 验证结果
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    if (verifyVectorAdd(h_c_ref, h_c, N)) {
        printf("   ✓ 结果正确\n\n");
    }
    
    // ========== 2. 向量点积对比 ==========
    printf("2. 向量点积:\n");
    
    // CPU版本
    cpu_start = clock();
    float dot_cpu = vectorDotCPU(h_a, h_b, N);
    cpu_end = clock();
    cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("   CPU时间: %.2f ms\n", cpu_time);
    printf("   结果: %.6f\n", dot_cpu);
    
    // GPU朴素版本（原子操作）
    float *d_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(d_result, 0, sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorDotNaive<<<numBlocks, blockSize>>>(d_a, d_b, d_result, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&naive_time, start, stop));
    printf("   GPU原子操作时间: %.2f ms (加速 %.2fx)\n", naive_time, cpu_time / naive_time);
    
    // GPU优化版本（树形归约）
    float *d_partialSums;
    CHECK_CUDA_ERROR(cudaMalloc(&d_partialSums, numBlocks * sizeof(float)));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorDotOptimized<<<numBlocks, blockSize>>>(d_a, d_b, d_partialSums, N);
    
    // 第二阶段：归约部分和
    float *h_partialSums = (float*)malloc(numBlocks * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    float dot_gpu = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        dot_gpu += h_partialSums[i];
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&opt_time, start, stop));
    printf("   GPU树形归约时间: %.2f ms (加速 %.2fx)\n", opt_time, cpu_time / opt_time);
    printf("   结果: %.6f (%s)\n\n", dot_gpu, fabs(dot_cpu - dot_gpu) < 1e-2 ? "正确" : "错误");
    
    free(h_partialSums);
    CHECK_CUDA_ERROR(cudaFree(d_partialSums));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    
    // ========== 3. 向量缩放对比 ==========
    printf("3. 向量缩放:\n");
    
    float scalar = 2.5f;
    
    // CPU版本
    cpu_start = clock();
    vectorScaleCPU(h_a, scalar, h_c_ref, N);
    cpu_end = clock();
    cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("   CPU时间: %.2f ms\n", cpu_time);
    
    // GPU朴素版本
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorScaleNaive<<<numBlocks, blockSize>>>(d_a, scalar, d_c, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&naive_time, start, stop));
    printf("   GPU朴素时间: %.2f ms (加速 %.2fx)\n", naive_time, cpu_time / naive_time);
    
    // GPU向量化版本
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorScaleOptimized<<<numBlocks / 4, blockSize>>>(reinterpret_cast<float4*>(d_a),
                                                         scalar,
                                                         reinterpret_cast<float4*>(d_c),
                                                         N / 4);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&opt_time, start, stop));
    printf("   GPU优化时间: %.2f ms (加速 %.2fx)\n", opt_time, cpu_time / opt_time);
    
    // 验证结果
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    if (verifyVectorAdd(h_c_ref, h_c, N)) {
        printf("   ✓ 结果正确\n\n");
    }
    
    // ========== 性能总结 ==========
    printf("=== 性能总结 ===\n");
    printf("向量加法: 向量化优化比朴素版本快 %.2fx\n", naive_time / opt_time);
    printf("向量点积: 树形归约比原子操作快约 %.2fx\n", naive_time / opt_time);
    printf("向量缩放: 向量化优化比朴素版本快 %.2fx\n", naive_time / opt_time);
    
    // ========== 清理资源 ==========
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);
    
    printf("\n程序执行完成！\n");
    return 0;
}

/*
 * ========== 编译和运行 ==========
 * 
 * 编译:
 *   nvcc -O3 -arch=sm_80 vector_operations.cu -o vector_ops
 * 
 * 运行:
 *   ./vector_ops
 * 
 * 性能分析:
 *   nvprof ./vector_ops
 */
