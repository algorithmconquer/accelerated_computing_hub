/*
 * Day 2 - 矩阵乘法优化
 * 
 * 目标：通过矩阵乘法掌握GPU性能优化技巧
 * 
 * 学习要点：
 * 1. CPU vs GPU性能对比
 * 2. 全局内存访问优化
 * 3. 共享内存使用
 * 4. 分块（Tiling）技术
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 矩阵尺寸
#define M 1024
#define K 1024
#define N 1024

// CPU版本：基准
void matmulCPU(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// GPU版本1：朴素实现
__global__ void matmulNaive(float *A, float *B, float *C, 
                             int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

// GPU版本2：使用共享内存（分块矩阵乘法）
#define TILE_SIZE 32

__global__ void matmulShared(float *A, float *B, float *C,
                              int m, int k, int n) {
    // 共享内存块
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // 计算全局坐标
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 分块循环
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载数据到共享内存
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        
        // 边界检查
        As[threadIdx.y][threadIdx.x] = (aRow < m && aCol < k) ? 
                                        A[aRow * k + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < k && bCol < n) ? 
                                        B[bRow * n + bCol] : 0.0f;
        
        __syncthreads();  // 确保数据已加载
        
        // 计算部分积
        for (int l = 0; l < TILE_SIZE; l++) {
            sum += As[threadIdx.y][l] * Bs[l][threadIdx.x];
        }
        
        __syncthreads();  // 确保计算完成
    }
    
    // 写回结果
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// 初始化矩阵
void initMatrix(float *mat, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = value;
    }
}

// 验证结果
bool verifyResult(float *C1, float *C2, int m, int n, float epsilon = 1e-3) {
    for (int i = 0; i < m * n; i++) {
        if (fabs(C1[i] - C2[i]) > epsilon) {
            printf("错误: index %d, 值 %.6f vs %.6f\n", 
                   i, C1[i], C2[i]);
            return false;
        }
    }
    return true;
}

// 性能测试
void benchmark() {
    printf("\n========================================\n");
    printf("矩阵乘法性能测试\n");
    printf("矩阵尺寸: A(%dx%d) × B(%dx%d) = C(%dx%d)\n", M, K, K, N, M, N);
    printf("========================================\n\n");
    
    // 分配主机内存
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C_cpu = (float*)malloc(M * N * sizeof(float));
    float *h_C_naive = (float*)malloc(M * N * sizeof(float));
    float *h_C_shared = (float*)malloc(M * N * sizeof(float));
    
    // 初始化数据
    printf("初始化矩阵...\n");
    initMatrix(h_A, M, K, 1.0f);
    initMatrix(h_B, K, N, 1.0f);
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 创建计时事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;
    
    // ===== CPU版本 =====
    printf("\n[1] CPU版本\n");
    clock_t cpu_start = clock();
    matmulCPU(h_A, h_B, h_C_cpu, M, K, N);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("  时间: %.3f ms\n", cpu_time);
    
    // ===== GPU朴素版本 =====
    printf("\n[2] GPU朴素版本\n");
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    
    cudaEventRecord(start);
    matmulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    cudaMemcpy(h_C_naive, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("  时间: %.3f ms\n", time);
    printf("  加速比: %.2fx\n", cpu_time / time);
    printf("  验证: %s\n", verifyResult(h_C_cpu, h_C_naive, M, N) ? "✓" : "✗");
    
    // ===== GPU共享内存版本 =====
    printf("\n[3] GPU共享内存版本\n");
    dim3 blockSize2(TILE_SIZE, TILE_SIZE);
    dim3 gridSize2((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEventRecord(start);
    matmulShared<<<gridSize2, blockSize2>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    cudaMemcpy(h_C_shared, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("  时间: %.3f ms\n", time);
    printf("  加速比: %.2fx (vs CPU)\n", cpu_time / time);
    printf("  验证: %s\n", verifyResult(h_C_cpu, h_C_shared, M, N) ? "✓" : "✗");
    
    // 性能分析
    printf("\n========================================\n");
    printf("性能分析\n");
    printf("========================================\n");
    
    // 计算GFLOPS
    double flops = 2.0 * M * K * N;  // 每个元素需要K次乘法和K次加法
    printf("总浮点运算: %.2f GFLOPS\n", flops / 1e9);
    printf("\nCPU版本:\n");
    printf("  性能: %.2f GFLOPS\n", flops / (cpu_time * 1e6));
    printf("\nGPU朴素版本:\n");
    printf("  性能: %.2f GFLOPS\n", flops / (time * 1e6));
    printf("\nGPU共享内存版本:\n");
    printf("  性能: %.2f GFLOPS\n", flops / (time * 1e6));
    
    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_naive);
    free(h_C_shared);
}

int main() {
    benchmark();
    printf("\n========================================\n");
    printf("测试完成!\n");
    printf("========================================\n");
    return 0;
}
