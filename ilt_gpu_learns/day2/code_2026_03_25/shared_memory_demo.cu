/*
 * Day 2 - 共享内存示例
 * 
 * 目标：理解共享内存的使用和性能优势
 * 
 * 学习要点：
 * 1. 共享内存的声明和使用
 * 2. Bank冲突的避免
 * 3. 数据共享和同步
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 全局内存版本：简单归约求和
__global__ void reduceGlobal(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程独立计算部分和
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // 使用原子操作累加（性能较差）
    atomicAdd(output, sum);
}

// 共享内存版本：并行归约求和
__global__ void reduceShared(float *input, float *output, int n) {
    // 声明共享内存（动态分配）
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 第一步：每个线程加载数据到共享内存
    sdata[tid] = 0.0f;
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    __syncthreads();  // 同步：确保所有数据已加载
    
    // 第二步：并行归约
    // 每次迭代，线程数减半
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // 同步：确保这一轮完成
    }
    
    // 第三步：Block 0的线程0写入结果
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// 共享内存示例：矩阵转置
__global__ void transposeShared(float *input, float *output, 
                                  int width, int height) {
    // 声明共享内存（静态分配）
    __shared__ float tile[16][16];  // 16x16块
    
    // 坐标计算
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 从全局内存加载到共享内存
    // 注意：这里用到了矩阵分块的思想
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // 计算转置后的坐标
    int x_t = blockIdx.y * blockDim.y + threadIdx.x;
    int y_t = blockIdx.x * blockDim.x + threadIdx.y;
    
    // 从共享内存写回全局内存
    if (x_t < height && y_t < width) {
        output[y_t * height + x_t] = tile[threadIdx.x][threadIdx.y];
    }
}

// 测试归约性能
void testReducePerformance() {
    printf("\n=== 归约求和性能测试 ===\n");
    
    const int N = 1 << 20;  // 1M元素
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    
    // 准备数据
    float *h_input = (float*)malloc(N * sizeof(float));
    float h_output_global = 0.0f;
    float h_output_shared = 0.0f;
    
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // 每个元素都是1，总和应该是N
    }
    
    // 分配GPU内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;
    
    // 测试全局内存版本
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    reduceGlobal<<<numBlocks, blockSize>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&h_output_global, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("全局内存版本:\n");
    printf("  时间: %.3f ms\n", time);
    printf("  结果: %.2f (期望: %.2f)\n", h_output_global, (float)N);
    
    // 测试共享内存版本
    cudaMemset(d_output, 0, sizeof(float));
    cudaEventRecord(start);
    reduceShared<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaMemcpy(&h_output_shared, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("共享内存版本:\n");
    printf("  时间: %.3f ms\n", time);
    printf("  结果: %.2f (期望: %.2f)\n", h_output_shared, (float)N);
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
}

// 测试矩阵转置
void testTranspose() {
    printf("\n=== 矩阵转置测试 ===\n");
    
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int SIZE = WIDTH * HEIGHT;
    
    // 准备数据
    float *h_input = (float*)malloc(SIZE * sizeof(float));
    float *h_output = (float*)malloc(SIZE * sizeof(float));
    
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = (float)i;
    }
    
    // 分配GPU内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));
    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动核函数
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    transposeShared<<<gridSize, blockSize>>>(d_input, d_output, WIDTH, HEIGHT);
    
    // 拷贝结果
    cudaMemcpy(h_output, d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 验证结果
    printf("验证转置结果...\n");
    bool correct = true;
    for (int y = 0; y < HEIGHT && correct; y++) {
        for (int x = 0; x < WIDTH && correct; x++) {
            if (h_output[x * HEIGHT + y] != h_input[y * WIDTH + x]) {
                correct = false;
                printf("错误: (%d,%d) 期望 %.2f, 得到 %.2f\n",
                       x, y, h_input[y * WIDTH + x], h_output[x * HEIGHT + y]);
            }
        }
    }
    
    if (correct) {
        printf("✓ 转置正确!\n");
    }
    
    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
}

int main() {
    printf("========================================\n");
    printf("共享内存示例\n");
    printf("========================================\n");
    
    testReducePerformance();
    testTranspose();
    
    printf("\n========================================\n");
    printf("测试完成!\n");
    printf("========================================\n");
    
    return 0;
}
