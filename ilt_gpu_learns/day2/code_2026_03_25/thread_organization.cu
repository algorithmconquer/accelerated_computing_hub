/*
 * Day 2 - 线程组织可视化示例
 * 
 * 目标：理解CUDA线程层次结构
 * 
 * 学习要点：
 * 1. Grid和Block的维度设置
 * 2. 线程索引的计算方法
 * 3. 一维和二维网格的使用场景
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 一维网格：处理数组
__global__ void processArray(float *data, int n) {
    // 计算全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查（重要！）
    if (idx < n) {
        data[idx] = idx * 2.0f;  // 简单的赋值操作
    }
}

// 二维网格：处理图像
__global__ void processImage(float *image, int width, int height) {
    // 计算二维坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 边界检查
    if (x < width && y < height) {
        int idx = y * width + x;  // 行优先存储
        image[idx] = (x + y) / (float)(width + height);
    }
}

// 打印线程块信息
__global__ void printBlockInfo() {
    // 每个Block的第一个线程打印信息
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Block(%d,%d): blockDim=(%d,%d), gridDim=(%d,%d)\n",
               blockIdx.x, blockIdx.y,
               blockDim.x, blockDim.y,
               gridDim.x, gridDim.y);
    }
}

// 测试一维网格
void test1DGrid() {
    printf("\n=== 测试一维网格 ===\n");
    
    const int N = 1000;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;
    
    printf("数据大小: %d\n", N);
    printf("块大小: %d\n", blockSize);
    printf("块数量: %d\n", numBlocks);
    printf("总线程数: %d\n\n", numBlocks * blockSize);
    
    // 分配内存
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    
    // 启动核函数
    processArray<<<numBlocks, blockSize>>>(d_data, N);
    
    // 拷贝结果回主机
    float *h_data = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 打印前10个元素
    printf("前10个元素:\n");
    for (int i = 0; i < 10; i++) {
        printf("  data[%d] = %.2f\n", i, h_data[i]);
    }
    
    // 清理
    cudaFree(d_data);
    free(h_data);
}

// 测试二维网格
void test2DGrid() {
    printf("\n=== 测试二维网格 ===\n");
    
    const int WIDTH = 640;
    const int HEIGHT = 480;
    
    // 块大小：16x16 = 256线程/块（常用配置）
    dim3 blockSize(16, 16);
    
    // 网格大小：向上取整
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    printf("图像大小: %d x %d = %d 像素\n", WIDTH, HEIGHT, WIDTH * HEIGHT);
    printf("块大小: %d x %d = %d 线程/块\n", 
           blockSize.x, blockSize.y, blockSize.x * blockSize.y);
    printf("网格大小: %d x %d = %d 块\n",
           gridSize.x, gridSize.y, gridSize.x * gridSize.y);
    printf("总线程数: %d\n\n",
           gridSize.x * gridSize.y * blockSize.x * blockSize.y);
    
    // 分配内存
    float *d_image;
    cudaMalloc(&d_image, WIDTH * HEIGHT * sizeof(float));
    
    // 启动核函数
    processImage<<<gridSize, blockSize>>>(d_image, WIDTH, HEIGHT);
    
    // 拷贝结果
    float *h_image = (float*)malloc(WIDTH * HEIGHT * sizeof(float));
    cudaMemcpy(h_image, d_image, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 打印部分结果
    printf("部分像素值:\n");
    for (int y = 0; y < 5; y++) {
        for (int x = 0; x < 5; x++) {
            printf("%.2f ", h_image[y * WIDTH + x]);
        }
        printf("\n");
    }
    
    // 清理
    cudaFree(d_image);
    free(h_image);
}

// 测试线程块信息
void testBlockInfo() {
    printf("\n=== 测试线程块信息 ===\n");
    
    dim3 blockSize(4, 4);
    dim3 gridSize(2, 2);
    
    printf("启动配置: gridSize=(%d,%d), blockSize=(%d,%d)\n\n",
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    printBlockInfo<<<gridSize, blockSize>>>();
    cudaDeviceSynchronize();
}

int main() {
    printf("========================================\n");
    printf("CUDA线程组织可视化\n");
    printf("========================================\n");
    
    test1DGrid();
    test2DGrid();
    testBlockInfo();
    
    printf("\n========================================\n");
    printf("测试完成!\n");
    printf("========================================\n");
    
    return 0;
}
