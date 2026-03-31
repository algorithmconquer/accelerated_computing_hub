/*
 * pinned_memory_demo.cu
 * 固定内存演示
 * 
 * 功能：
 * 1. 对比固定内存与可分页内存
 * 2. 测量传输性能差异
 * 3. 演示异步传输
 * 4. 内存使用分析
 * 
 * 编译: nvcc -O3 -arch=sm_80 pinned_memory_demo.cu -o pinned_memory_demo
 * 运行: ./pinned_memory_demo
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/sysctl.h>

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define DATA_SIZE_MB 128
#define TEST_ITERATIONS 10

// 获取系统总内存（macOS）
size_t get_system_memory() {
    int mib[2];
    int64_t physical_memory;
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    size_t length = sizeof(int64_t);
    sysctl(mib, 2, &physical_memory, &length, NULL, 0);
    return physical_memory;
}

// 测试可分页内存传输
float test_pageable_memory(float* h_data, float* d_data, size_t size, 
                           int iterations) {
    /*
     * 使用可分页内存（malloc）进行同步传输
     * 每次传输需要先"钉住"内存，增加开销
     */
    printf("测试可分页内存 (malloc + 同步传输):\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float total_time = 0.0f;
    
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        // 同步传输H2D
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_time += ms;
    }
    
    float avg_time = total_time / iterations;
    float bandwidth = (size / 1e6) / (avg_time / 1e3);  // GB/s
    
    printf("  平均传输时间: %.2f ms\n", avg_time);
    printf("  带宽: %.2f GB/s\n", bandwidth);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return avg_time;
}

// 测试固定内存同步传输
float test_pinned_sync(float* h_data, float* d_data, size_t size, 
                       int iterations) {
    /*
     * 使用固定内存（cudaMallocHost）进行同步传输
     * DMA可以直接访问，但仍是同步操作
     */
    printf("\n测试固定内存 (cudaMallocHost + 同步传输):\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float total_time = 0.0f;
    
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        // 同步传输H2D
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_time += ms;
    }
    
    float avg_time = total_time / iterations;
    float bandwidth = (size / 1e6) / (avg_time / 1e3);  // GB/s
    
    printf("  平均传输时间: %.2f ms\n", avg_time);
    printf("  带宽: %.2f GB/s\n", bandwidth);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return avg_time;
}

// 测试固定内存异步传输
float test_pinned_async(float* h_data, float* d_data, size_t size, 
                        int iterations) {
    /*
     * 使用固定内存进行异步传输
     * 传输与计算可以重叠
     */
    printf("\n测试固定内存 (cudaMallocHost + 异步传输):\n");
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float total_time = 0.0f;
    
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        
        // 异步传输H2D
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, size, 
                                   cudaMemcpyHostToDevice, stream));
        
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_time += ms;
    }
    
    float avg_time = total_time / iterations;
    float bandwidth = (size / 1e6) / (avg_time / 1e3);  // GB/s
    
    printf("  平均传输时间: %.2f ms\n", avg_time);
    printf("  带宽: %.2f GB/s\n", bandwidth);
    
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return avg_time;
}

// 测试双向并发传输
float test_bidirectional_transfer(float* h_data1, float* h_data2,
                                   float* d_data1, float* d_data2,
                                   size_t size, int iterations) {
    /*
     * 测试同时进行H2D和D2H传输
     * 需要固定内存支持
     */
    printf("\n测试双向并发传输 (固定内存):\n");
    
    cudaStream_t stream_h2d, stream_d2h;
    CUDA_CHECK(cudaStreamCreate(&stream_h2d));
    CUDA_CHECK(cudaStreamCreate(&stream_d2h));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float total_time = 0.0f;
    
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        // H2D传输
        CUDA_CHECK(cudaMemcpyAsync(d_data1, h_data1, size, 
                                   cudaMemcpyHostToDevice, stream_h2d));
        
        // 同时进行D2H传输
        CUDA_CHECK(cudaMemcpyAsync(h_data2, d_data2, size, 
                                   cudaMemcpyDeviceToHost, stream_d2h));
        
        // 同步两个流
        CUDA_CHECK(cudaStreamSynchronize(stream_h2d));
        CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_time += ms;
    }
    
    float avg_time = total_time / iterations;
    float bandwidth = (2 * size / 1e6) / (avg_time / 1e3);  // 双向带宽
    
    printf("  平均传输时间: %.2f ms\n", avg_time);
    printf("  双向数据量: %zu MB\n", 2 * size / (1024 * 1024));
    printf("  双向带宽: %.2f GB/s\n", bandwidth);
    
    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_d2h));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return avg_time;
}

int main() {
    printf("=== 固定内存演示 ===\n\n");
    
    // 显示GPU信息
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n", prop.name);
    printf("PCIe带宽: PCIe Gen%d\n\n", 
           prop.pciBusID ? 3 : 4);  // 简化判断
    
    // 数据大小
    size_t size = DATA_SIZE_MB * 1024 * 1024;
    int iterations = TEST_ITERATIONS;
    
    printf("测试配置:\n");
    printf("  数据大小: %d MB\n", DATA_SIZE_MB);
    printf("  测试次数: %d\n\n", iterations);
    
    // ========== 测试可分页内存 ==========
    float* h_pageable = (float*)malloc(size);
    for (size_t i = 0; i < size / sizeof(float); i++) {
        h_pageable[i] = (float)i;
    }
    
    float* d_data1;
    CUDA_CHECK(cudaMalloc(&d_data1, size));
    
    float time_pageable = test_pageable_memory(h_pageable, d_data1, 
                                                size, iterations);
    
    // ========== 测试固定内存同步传输 ==========
    float* h_pinned1;
    CUDA_CHECK(cudaMallocHost(&h_pinned1, size));
    for (size_t i = 0; i < size / sizeof(float); i++) {
        h_pinned1[i] = (float)i;
    }
    
    float time_pinned_sync = test_pinned_sync(h_pinned1, d_data1, 
                                               size, iterations);
    
    // ========== 测试固定内存异步传输 ==========
    float time_pinned_async = test_pinned_async(h_pinned1, d_data1, 
                                                 size, iterations);
    
    // ========== 测试双向并发传输 ==========
    float* h_pinned2;
    float* d_data2;
    CUDA_CHECK(cudaMallocHost(&h_pinned2, size));
    CUDA_CHECK(cudaMalloc(&d_data2, size));
    
    float time_bidirectional = test_bidirectional_transfer(
        h_pinned1, h_pinned2, d_data1, d_data2, size, iterations);
    
    // ========== 性能对比 ==========
    printf("\n========================================\n");
    printf("性能对比总结:\n");
    printf("========================================\n");
    printf("方法                          时间(ms)    加速比\n");
    printf("----------------------------------------\n");
    printf("可分页内存(同步)             %7.2f     1.00x\n", time_pageable);
    printf("固定内存(同步)               %7.2f     %.2fx\n", 
           time_pinned_sync, time_pageable / time_pinned_sync);
    printf("固定内存(异步)               %7.2f     %.2fx\n", 
           time_pinned_async, time_pageable / time_pinned_async);
    printf("双向并发传输                 %7.2f     %.2fx\n", 
           time_bidirectional, time_pageable / time_bidirectional);
    printf("========================================\n\n");
    
    // ========== 内存使用分析 ==========
    size_t system_mem = get_system_memory();
    size_t pinned_usage = 2 * size;
    
    printf("内存使用分析:\n");
    printf("  系统总内存: %.2f GB\n", system_mem / 1e9);
    printf("  固定内存占用: %.2f MB\n", pinned_usage / 1e6);
    printf("  占比: %.2f%%\n\n", 
           (float)pinned_usage / system_mem * 100);
    
    // ========== 最佳实践建议 ==========
    printf("固定内存最佳实践:\n");
    printf("  ✓ 适用于频繁传输的大块数据\n");
    printf("  ✓ 异步传输必须使用固定内存\n");
    printf("  ✓ 可以实现双向并发传输\n");
    printf("  ✗ 不要过度使用（占用物理RAM）\n");
    printf("  ✗ 小数据块收益不明显\n");
    printf("  ✗ 分配/释放开销较大\n\n");
    
    // 清理内存
    free(h_pageable);
    CUDA_CHECK(cudaFreeHost(h_pinned1));
    CUDA_CHECK(cudaFreeHost(h_pinned2));
    CUDA_CHECK(cudaFree(d_data1));
    CUDA_CHECK(cudaFree(d_data2));
    
    return 0;
}
