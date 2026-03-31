/*
 * stream_comparison.cu
 * CUDA流性能对比：单流 vs 多流
 * 
 * 功能：
 * 1. 实现单流执行版本
 * 2. 实现2、4、8流并发版本
 * 3. 测量并对比性能
 * 4. 计算加速比
 * 
 * 编译: nvcc -O3 -arch=sm_80 stream_comparison.cu -o stream_comparison
 * 运行: ./stream_comparison
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

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

// 数据大小 (64 MB)
#define DATA_SIZE (64 * 1024 * 1024)
#define DATA_COUNT (DATA_SIZE / sizeof(float))

// 核函数：向量运算（模拟计算密集型任务）
__global__ void vector_compute(float* data, int n, int iterations) {
    /*
     * 每个线程处理一个元素，执行多次迭代运算
     * iterations参数控制计算复杂度
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        
        // 执行多次运算，增加计算强度
        for (int i = 0; i < iterations; i++) {
            val = val * 1.01f + 0.1f;
            val = sqrtf(val * val + 1.0f);
        }
        
        data[idx] = val;
    }
}

// 单流版本
float execute_single_stream(float* h_data, float* d_data, int n, int iterations) {
    /*
     * 单流执行流程：
     * 1. H2D传输
     * 2. 核函数执行
     * 3. D2H传输
     * 所有操作串行执行
     */
    
    // 创建事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 开始计时
    CUDA_CHECK(cudaEventRecord(start));
    
    // 1. H2D传输（同步）
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // 2. 核函数执行
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vector_compute<<<gridSize, blockSize>>>(d_data, n, iterations);
    CUDA_CHECK(cudaGetLastError());
    
    // 3. D2H传输（同步）
    CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 结束计时
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // 清理事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds;
}

// 多流版本
float execute_multi_stream(float* h_data, float* d_data, int n, 
                           int iterations, int num_streams) {
    /*
     * 多流执行流程：
     * 1. 创建多个流
     * 2. 将数据分块
     * 3. 每个流处理一个数据块
     * 4. 异步执行传输和计算
     * 5. 流水线并行
     */
    
    // 创建流
    cudaStream_t* streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // 创建事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 计算每个流处理的数据量
    int chunk_size = n / num_streams;
    
    // 开始计时
    CUDA_CHECK(cudaEventRecord(start));
    
    // 在多个流中并发执行
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        int current_size = (i == num_streams - 1) ? (n - offset) : chunk_size;
        
        // 1. 异步H2D传输
        CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset, 
                                   current_size * sizeof(float), 
                                   cudaMemcpyHostToDevice, streams[i]));
        
        // 2. 核函数执行
        int blockSize = 256;
        int gridSize = (current_size + blockSize - 1) / blockSize;
        vector_compute<<<gridSize, blockSize, 0, streams[i]>>>(
            d_data + offset, current_size, iterations);
        
        // 3. 异步D2H传输
        CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset, 
                                   current_size * sizeof(float), 
                                   cudaMemcpyDeviceToHost, streams[i]));
    }
    
    // 同步所有流
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // 结束计时
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // 清理
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds;
}

// 运行多次取平均
float benchmark_single(float* h_data, float* d_data, int n, 
                       int iterations, int runs) {
    float total_time = 0.0f;
    for (int i = 0; i < runs; i++) {
        total_time += execute_single_stream(h_data, d_data, n, iterations);
    }
    return total_time / runs;
}

float benchmark_multi(float* h_data, float* d_data, int n, 
                      int iterations, int num_streams, int runs) {
    float total_time = 0.0f;
    for (int i = 0; i < runs; i++) {
        total_time += execute_multi_stream(h_data, d_data, n, iterations, num_streams);
    }
    return total_time / runs;
}

int main() {
    printf("=== CUDA流性能对比测试 ===\n\n");
    
    // 显示GPU信息
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %.2f GB\n\n", prop.totalGlobalMem / 1e9);
    
    // 参数设置
    int n = DATA_COUNT;
    int iterations = 100;  // 计算迭代次数
    int runs = 10;         // 每个测试运行次数
    
    printf("数据大小: %d MB\n", DATA_SIZE / (1024 * 1024));
    printf("元素数量: %d\n", n);
    printf("计算迭代: %d\n", iterations);
    printf("测试次数: %d\n\n", runs);
    
    // 分配主机内存
    float* h_data = (float*)malloc(n * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "主机内存分配失败\n");
        return 1;
    }
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i * 0.001f;
    }
    
    // 分配设备内存
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(float)));
    
    // 测试单流
    printf("测试单流版本...\n");
    float single_time = benchmark_single(h_data, d_data, n, iterations, runs);
    printf("  平均时间: %.2f ms\n\n", single_time);
    
    // 测试多流（2、4、8个流）
    int stream_counts[] = {2, 4, 8};
    float best_speedup = 1.0f;
    int best_num_streams = 1;
    
    for (int i = 0; i < 3; i++) {
        int num_streams = stream_counts[i];
        printf("测试%d流版本...\n", num_streams);
        
        float multi_time = benchmark_multi(h_data, d_data, n, 
                                           iterations, num_streams, runs);
        float speedup = single_time / multi_time;
        
        printf("  平均时间: %.2f ms\n", multi_time);
        printf("  加速比: %.2fx\n\n", speedup);
        
        if (speedup > best_speedup) {
            best_speedup = speedup;
            best_num_streams = num_streams;
        }
    }
    
    // 总结
    printf("========================================\n");
    printf("性能总结:\n");
    printf("  单流时间: %.2f ms (基准)\n", single_time);
    printf("  最佳配置: %d 个流\n", best_num_streams);
    printf("  最佳加速比: %.2fx\n", best_speedup);
    printf("  性能提升: %.1f%%\n", (best_speedup - 1.0f) * 100);
    printf("========================================\n\n");
    
    // 性能分析
    printf("性能分析:\n");
    printf("  1. 多流通过传输-计算重叠实现加速\n");
    printf("  2. 最佳流数量取决于数据大小和GPU架构\n");
    printf("  3. 流过多会导致资源竞争，反而降低性能\n");
    printf("  4. 建议: 数据量 > 64MB时使用4-8个流\n");
    
    // 清理内存
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    
    return 0;
}
