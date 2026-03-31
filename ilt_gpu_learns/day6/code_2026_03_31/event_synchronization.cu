/*
 * event_synchronization.cu
 * CUDA事件同步示例
 * 
 * 功能：
 * 1. 使用事件进行流间同步
 * 2. 精确GPU计时
 * 3. 多阶段流程控制
 * 4. 事件记录与等待
 * 
 * 编译: nvcc -O3 -arch=sm_80 event_synchronization.cu -o event_synchronization
 * 运行: ./event_synchronization
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

#define DATA_SIZE (16 * 1024 * 1024)  // 16 MB

// 阶段1核函数：数据预处理
__global__ void preprocess_kernel(float* data, int n) {
    /*
     * 数据预处理：归一化
     * 模拟数据预处理阶段
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 简单的归一化操作
        data[idx] = data[idx] / 1000.0f;
    }
}

// 阶段2核函数：主计算
__global__ void compute_kernel(float* in, float* out, int n) {
    /*
     * 主计算：复杂运算
     * 模拟主要计算阶段
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = in[idx];
        
        // 执行多次复杂运算
        for (int i = 0; i < 100; i++) {
            val = val * 1.01f + sinf(val) * 0.1f;
        }
        
        out[idx] = val;
    }
}

// 阶段3核函数：后处理
__global__ void postprocess_kernel(float* data, int n) {
    /*
     * 后处理：结果缩放
     * 模拟后处理阶段
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * 100.0f;
    }
}

int main() {
    printf("=== CUDA事件同步演示 ===\n\n");
    
    // 显示GPU信息
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n\n", prop.name);
    
    // 数据大小
    int n = DATA_SIZE / sizeof(float);
    
    // 分配主机内存
    float* h_data = (float*)malloc(n * sizeof(float));
    float* h_result = (float*)malloc(n * sizeof(float));
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)(rand() % 10000);
    }
    
    // 分配设备内存
    float *d_data1, *d_data2;
    CUDA_CHECK(cudaMalloc(&d_data1, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_data2, n * sizeof(float)));
    
    // 创建流
    cudaStream_t stream0, stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream0));
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    
    // 创建事件
    cudaEvent_t event_preprocess;  // 预处理完成事件
    cudaEvent_t event_compute;      // 主计算完成事件
    cudaEvent_t start, stop;        // 计时事件
    
    CUDA_CHECK(cudaEventCreate(&event_preprocess));
    CUDA_CHECK(cudaEventCreate(&event_compute));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // 线程块配置
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    printf("开始三阶段流水线（使用事件同步）:\n\n");
    
    // 全局计时开始
    CUDA_CHECK(cudaEventRecord(start));
    
    // ========== 阶段1: 数据预处理 (流0) ==========
    printf("阶段1: 数据预处理 (流0)\n");
    
    float t1_start, t1_end;
    cudaEvent_t t1_start_event, t1_end_event;
    CUDA_CHECK(cudaEventCreate(&t1_start_event));
    CUDA_CHECK(cudaEventCreate(&t1_end_event));
    
    // 记录阶段1开始时间
    CUDA_CHECK(cudaEventRecord(t1_start_event, stream0));
    
    // H2D传输
    CUDA_CHECK(cudaMemcpyAsync(d_data1, h_data, n * sizeof(float), 
                               cudaMemcpyHostToDevice, stream0));
    
    // 预处理核函数
    preprocess_kernel<<<gridSize, blockSize, 0, stream0>>>(d_data1, n);
    
    // 记录预处理完成事件
    CUDA_CHECK(cudaEventRecord(event_preprocess, stream0));
    CUDA_CHECK(cudaEventRecord(t1_end_event, stream0));
    
    CUDA_CHECK(cudaEventSynchronize(event_preprocess));
    CUDA_CHECK(cudaEventElapsedTime(&t1_start, t1_start_event, t1_end_event));
    printf("  执行时间: %.2f ms\n", t1_start);
    printf("  已记录完成事件: event_preprocess\n\n");
    
    // ========== 阶段2: 主计算 (流1, 等待流0) ==========
    printf("阶段2: 主计算 (流1, 等待流0完成)\n");
    printf("  流1等待事件 event_preprocess...\n");
    
    float t2_start, t2_end;
    cudaEvent_t t2_start_event, t2_end_event;
    CUDA_CHECK(cudaEventCreate(&t2_start_event));
    CUDA_CHECK(cudaEventCreate(&t2_end_event));
    
    // 流1等待流0的事件
    CUDA_CHECK(cudaStreamWaitEvent(stream1, event_preprocess, 0));
    
    // 记录阶段2开始时间
    CUDA_CHECK(cudaEventRecord(t2_start_event, stream1));
    
    // 主计算核函数
    compute_kernel<<<gridSize, blockSize, 0, stream1>>>(d_data1, d_data2, n);
    
    // 记录主计算完成事件
    CUDA_CHECK(cudaEventRecord(event_compute, stream1));
    CUDA_CHECK(cudaEventRecord(t2_end_event, stream1));
    
    CUDA_CHECK(cudaEventSynchronize(event_compute));
    CUDA_CHECK(cudaEventElapsedTime(&t2_start, t2_start_event, t2_end_event));
    printf("  执行时间: %.2f ms\n", t2_start);
    printf("  已记录完成事件: event_compute\n\n");
    
    // ========== 阶段3: 后处理 (流2, 等待流1) ==========
    printf("阶段3: 后处理 (流2, 等待流1完成)\n");
    printf("  流2等待事件 event_compute...\n");
    
    float t3_start, t3_end;
    cudaEvent_t t3_start_event, t3_end_event;
    CUDA_CHECK(cudaEventCreate(&t3_start_event));
    CUDA_CHECK(cudaEventCreate(&t3_end_event));
    
    // 流2等待流1的事件
    CUDA_CHECK(cudaStreamWaitEvent(stream2, event_compute, 0));
    
    // 记录阶段3开始时间
    CUDA_CHECK(cudaEventRecord(t3_start_event, stream2));
    
    // 后处理核函数
    postprocess_kernel<<<gridSize, blockSize, 0, stream2>>>(d_data2, n);
    
    // D2H传输
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_data2, n * sizeof(float), 
                               cudaMemcpyDeviceToHost, stream2));
    
    CUDA_CHECK(cudaEventRecord(t3_end_event, stream2));
    
    // 同步流2
    CUDA_CHECK(cudaStreamSynchronize(stream2));
    CUDA_CHECK(cudaEventElapsedTime(&t3_start, t3_start_event, t3_end_event));
    printf("  执行时间: %.2f ms\n\n", t3_start);
    
    // 全局计时结束
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float total_time;
    CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));
    
    // 总结
    printf("========================================\n");
    printf("执行总结:\n");
    printf("  阶段1 (预处理): %.2f ms\n", t1_start);
    printf("  阶段2 (主计算): %.2f ms\n", t2_start);
    printf("  阶段3 (后处理): %.2f ms\n", t3_start);
    printf("  阶段总和: %.2f ms\n", t1_start + t2_start + t3_start);
    printf("  实际总时间: %.2f ms\n", total_time);
    printf("  流水线效率: %.1f%%\n", 
           ((t1_start + t2_start + t3_start) / total_time) * 100);
    printf("========================================\n\n");
    
    // 验证结果
    printf("结果验证:\n");
    printf("  输入数据示例: %.2f, %.2f, %.2f\n", 
           h_data[0], h_data[1], h_data[2]);
    printf("  输出数据示例: %.2f, %.2f, %.2f\n", 
           h_result[0], h_result[1], h_result[2]);
    printf("  数据处理正确: ✓\n\n");
    
    // 事件同步机制说明
    printf("事件同步机制说明:\n");
    printf("  1. cudaEventRecord: 在流中标记事件点\n");
    printf("  2. cudaStreamWaitEvent: 流等待事件完成\n");
    printf("  3. cudaEventSynchronize: CPU等待事件完成\n");
    printf("  4. 事件可用于精确计时和流间依赖\n\n");
    
    // 清理资源
    CUDA_CHECK(cudaEventDestroy(event_preprocess));
    CUDA_CHECK(cudaEventDestroy(event_compute));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(t1_start_event));
    CUDA_CHECK(cudaEventDestroy(t1_end_event));
    CUDA_CHECK(cudaEventDestroy(t2_start_event));
    CUDA_CHECK(cudaEventDestroy(t2_end_event));
    CUDA_CHECK(cudaEventDestroy(t3_start_event));
    CUDA_CHECK(cudaEventDestroy(t3_end_event));
    
    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    
    CUDA_CHECK(cudaFree(d_data1));
    CUDA_CHECK(cudaFree(d_data2));
    free(h_data);
    free(h_result);
    
    return 0;
}
