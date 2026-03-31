/*
 * pipeline_vector_add.cu
 * 流水线并行向量加法
 * 
 * 功能：
 * 1. 实现流水线式的向量运算
 * 2. H2D传输、计算、D2H传输并行执行
 * 3. 使用固定内存优化传输
 * 4. 可视化时间线
 * 
 * 编译: nvcc -O3 -arch=sm_80 pipeline_vector_add.cu -o pipeline_vector_add
 * 运行: ./pipeline_vector_add
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

// 配置参数
#define TOTAL_SIZE_MB 256
#define NUM_CHUNKS 4
#define ITERATIONS 50

// 核函数：向量运算
__global__ void vector_kernel(float* in, float* out, int n, int iterations) {
    /*
     * 模拟计算密集型任务
     * 每个线程处理一个元素，执行多次迭代
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = in[idx];
        
        for (int i = 0; i < iterations; i++) {
            val = val * 1.001f + 0.5f;
            val = sinf(val) + cosf(val);
        }
        
        out[idx] = val;
    }
}

// 时间线记录结构
struct TimelineEvent {
    const char* name;
    float start_ms;
    float duration_ms;
    int stream_id;
};

// 单流顺序执行
float sequential_execution(float* h_in, float* h_out, float* d_in, float* d_out,
                           int total_size, int iterations) {
    /*
     * 顺序执行：
     * H2D -> Compute -> D2H (全部完成后才进行下一块)
     */
    printf("\n=== 单流顺序执行 ===\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // H2D传输
    CUDA_CHECK(cudaMemcpy(d_in, h_in, total_size * sizeof(float), 
                          cudaMemcpyHostToDevice));
    
    // 计算
    int blockSize = 256;
    int gridSize = (total_size + blockSize - 1) / blockSize;
    vector_kernel<<<gridSize, blockSize>>>(d_in, d_out, total_size, iterations);
    CUDA_CHECK(cudaGetLastError());
    
    // D2H传输
    CUDA_CHECK(cudaMemcpy(h_out, d_out, total_size * sizeof(float), 
                          cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float total_time;
    CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));
    
    printf("  顺序执行时间: %.2f ms\n", total_time);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return total_time;
}

// 流水线并行执行
float pipeline_execution(float* h_in, float* h_out, float* d_in, float* d_out,
                         int total_size, int num_chunks, int iterations) {
    /*
     * 流水线并行：
     * 1. 数据分块
     * 2. 每个流处理一个块
     * 3. 传输-计算-传输流水线
     * 4. 重叠执行
     */
    printf("\n=== 流水线并行执行 (%d 个流) ===\n", num_chunks);
    
    int chunk_size = total_size / num_chunks;
    
    // 创建流和事件
    cudaStream_t* streams = new cudaStream_t[num_chunks];
    cudaEvent_t* events_h2d = new cudaEvent_t[num_chunks];
    cudaEvent_t* events_kernel = new cudaEvent_t[num_chunks];
    cudaEvent_t* events_d2h = new cudaEvent_t[num_chunks];
    
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&events_h2d[i]));
        CUDA_CHECK(cudaEventCreate(&events_kernel[i]));
        CUDA_CHECK(cudaEventCreate(&events_d2h[i]));
    }
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // 在每个流中执行流水线
    for (int i = 0; i < num_chunks; i++) {
        int offset = i * chunk_size;
        int current_size = (i == num_chunks - 1) ? 
                           (total_size - offset) : chunk_size;
        
        // 1. H2D传输
        CUDA_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset, 
                                   current_size * sizeof(float), 
                                   cudaMemcpyHostToDevice, streams[i]));
        CUDA_CHECK(cudaEventRecord(events_h2d[i], streams[i]));
        
        // 2. 核函数执行
        int blockSize = 256;
        int gridSize = (current_size + blockSize - 1) / blockSize;
        vector_kernel<<<gridSize, blockSize, 0, streams[i]>>>(
            d_in + offset, d_out + offset, current_size, iterations);
        CUDA_CHECK(cudaEventRecord(events_kernel[i], streams[i]));
        
        // 3. D2H传输
        CUDA_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset, 
                                   current_size * sizeof(float), 
                                   cudaMemcpyDeviceToHost, streams[i]));
        CUDA_CHECK(cudaEventRecord(events_d2h[i], streams[i]));
    }
    
    // 同步所有流
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float total_time;
    CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));
    
    // 分析时间线
    printf("\n时间线分析:\n");
    for (int i = 0; i < num_chunks; i++) {
        float h2d_time, kernel_time, d2h_time;
        
        CUDA_CHECK(cudaEventElapsedTime(&h2d_time, start, events_h2d[i]));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time, events_h2d[i], events_kernel[i]));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_time, events_kernel[i], events_d2h[i]));
        
        printf("  流%d: [H2D: %.1fms][Kernel: %.1fms][D2H: %.1fms]\n", 
               i, h2d_time, kernel_time, d2h_time);
    }
    
    printf("\n  流水线总时间: %.2f ms\n", total_time);
    
    // 清理
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaEventDestroy(events_h2d[i]));
        CUDA_CHECK(cudaEventDestroy(events_kernel[i]));
        CUDA_CHECK(cudaEventDestroy(events_d2h[i]));
    }
    delete[] streams;
    delete[] events_h2d;
    delete[] events_kernel;
    delete[] events_d2h;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return total_time;
}

int main() {
    printf("=== 流水线并行向量运算 ===\n\n");
    
    // 显示GPU信息
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n", prop.name);
    printf("全局内存: %.2f GB\n\n", prop.totalGlobalMem / 1e9);
    
    // 参数设置
    int total_size = (TOTAL_SIZE_MB * 1024 * 1024) / sizeof(float);
    int num_chunks = NUM_CHUNKS;
    int iterations = ITERATIONS;
    
    printf("配置参数:\n");
    printf("  数据总量: %d MB\n", TOTAL_SIZE_MB);
    printf("  分块数: %d\n", num_chunks);
    printf("  每块大小: %d MB\n", TOTAL_SIZE_MB / num_chunks);
    printf("  计算迭代: %d 次\n\n", iterations);
    
    // 分配固定内存（Pinned Memory）
    float* h_in;
    float* h_out;
    CUDA_CHECK(cudaMallocHost(&h_in, total_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_out, total_size * sizeof(float)));
    
    // 初始化数据
    printf("初始化数据...\n");
    for (int i = 0; i < total_size; i++) {
        h_in[i] = (float)i * 0.001f;
        h_out[i] = 0.0f;
    }
    
    // 分配设备内存
    float* d_in;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_in, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_size * sizeof(float)));
    
    // 测试顺序执行
    float seq_time = sequential_execution(h_in, h_out, d_in, d_out, 
                                          total_size, iterations);
    
    // 重置输出
    for (int i = 0; i < total_size; i++) {
        h_out[i] = 0.0f;
    }
    
    // 测试流水线执行
    float pipe_time = pipeline_execution(h_in, h_out, d_in, d_out, 
                                         total_size, num_chunks, iterations);
    
    // 性能对比
    printf("\n========================================\n");
    printf("性能对比:\n");
    printf("  顺序执行: %.2f ms\n", seq_time);
    printf("  流水线执行: %.2f ms\n", pipe_time);
    printf("  加速比: %.2fx\n", seq_time / pipe_time);
    printf("  时间节省: %.1f%%\n", 
           (1.0f - pipe_time / seq_time) * 100);
    printf("========================================\n\n");
    
    // 分析
    printf("流水线优化分析:\n");
    printf("  1. 传输和计算重叠执行\n");
    printf("  2. 多个流并发处理不同数据块\n");
    printf("  3. 固定内存提供更高的传输带宽\n");
    printf("  4. 最佳分块数取决于GPU和PCIe带宽\n\n");
    
    // 验证结果
    printf("结果验证:\n");
    bool correct = true;
    for (int i = 0; i < min(100, total_size); i++) {
        if (h_out[i] == 0.0f) {
            correct = false;
            break;
        }
    }
    printf("  数据计算正确: %s\n", correct ? "✓" : "✗");
    
    // 清理内存
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    
    return 0;
}
