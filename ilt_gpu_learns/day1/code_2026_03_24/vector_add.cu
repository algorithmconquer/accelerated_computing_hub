/*
 * CUDA向量加法示例
 * 展示基本的CUDA程序结构
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA核函数 - 在GPU上并行执行
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // 计算全局线程索引
    // blockIdx.x: 当前线程块在网格中的索引
    // blockDim.x: 每个线程块中的线程数
    // threadIdx.x: 当前线程在线程块中的索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查,防止越界访问
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

// CPU版本向量加法(用于对比)
void vectorAddCPU(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// 检查CUDA错误
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

int main() {
    printf("========================================\n");
    printf("CUDA向量加法示例\n");
    printf("========================================\n\n");
    
    // 向量大小
    int n = 10000000;  // 10M个元素
    size_t size = n * sizeof(float);
    
    printf("向量大小: %d 元素 (%.2f MB)\n\n", n, size / 1024.0 / 1024.0);
    
    // 分配主机内存(使用cudaMallocManaged可以简化内存管理)
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    float *h_c_cpu = (float*)malloc(size);  // CPU计算结果
    
    // 初始化数据
    printf("初始化数据...\n");
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }
    
    // 分配设备内存
    printf("分配GPU内存...\n");
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // ========== GPU计算 ==========
    printf("\n【GPU计算】\n");
    
    // 数据传输: Host → Device
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float memcpy_h2d_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&memcpy_h2d_time, start, stop));
    printf("数据传输 Host→Device: %.4f ms\n", memcpy_h2d_time);
    
    // 配置核函数启动参数
    int blockSize = 256;  // 每个线程块256个线程
    int numBlocks = (n + blockSize - 1) / blockSize;  // 计算需要的线程块数
    
    printf("线程块大小: %d\n", blockSize);
    printf("线程块数量: %d\n", numBlocks);
    printf("总线程数: %d\n", numBlocks * blockSize);
    
    // 启动核函数
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    CHECK_CUDA_ERROR(cudaGetLastError());  // 检查核函数启动错误
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());  // 等待核函数完成
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float kernel_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel_time, start, stop));
    printf("核函数执行时间: %.4f ms\n", kernel_time);
    
    // 数据传输: Device → Host
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float memcpy_d2h_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&memcpy_d2h_time, start, stop));
    printf("数据传输 Device→Host: %.4f ms\n", memcpy_d2h_time);
    
    float gpu_total_time = memcpy_h2d_time + kernel_time + memcpy_d2h_time;
    printf("GPU总时间: %.4f ms\n", gpu_total_time);
    
    // ========== CPU计算(对比) ==========
    printf("\n【CPU计算】\n");
    clock_t cpu_start = clock();
    vectorAddCPU(h_a, h_b, h_c_cpu, n);
    clock_t cpu_end = clock();
    
    float cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU计算时间: %.4f ms\n", cpu_time);
    
    // 计算加速比
    printf("\n【性能对比】\n");
    printf("加速比(GPU vs CPU): %.2f x\n", cpu_time / gpu_total_time);
    printf("加速比(仅核函数): %.2f x\n", cpu_time / kernel_time);
    
    // 验证结果正确性
    printf("\n【验证结果】\n");
    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        float error = fabs(h_c[i] - h_c_cpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    printf("最大误差: %.10f\n", max_error);
    
    if (max_error < 1e-5) {
        printf("✓ 结果验证通过!\n");
    } else {
        printf("✗ 结果验证失败!\n");
    }
    
    // 显示前5个结果
    printf("\n【示例结果(前5个)】\n");
    printf("索引    a          +    b          =    c(GPU)      c(CPU)\n");
    printf("-----   ----------     ----------     ----------     ----------\n");
    for (int i = 0; i < 5; i++) {
        printf("%5d   %10.6f   %10.6f   %10.6f   %10.6f\n", 
               i, h_a[i], h_b[i], h_c[i], h_c_cpu[i]);
    }
    
    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    printf("\n========================================\n");
    printf("程序执行完成!\n");
    printf("========================================\n");
    
    return 0;
}
