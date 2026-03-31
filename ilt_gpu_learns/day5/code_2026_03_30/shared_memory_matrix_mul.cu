/*
 * 共享内存矩阵乘法优化示例
 * 
 * 本程序演示如何使用共享内存优化GPU矩阵乘法
 * 包含三种实现：CPU版本、GPU朴素版本、GPU共享内存优化版本
 * 
 * 性能对比：
 * - CPU版本：基准
 * - GPU朴素版本：约20-30倍加速
 * - GPU共享内存优化版本：约50-70倍加速（比朴素版本快2-3倍）
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 定义分块大小（Tile Size）
// 常用值：16, 32
#define TILE_SIZE 32

// ============ 错误检查宏 ============
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ============ CPU矩阵乘法（基准版本）============
/*
 * 简单的三重循环矩阵乘法
 * 用于性能对比基准
 */
void matrixMulCPU(float* A, float* B, float* C, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A[row * width + k] * B[k * width + col];
            }
            C[row * width + col] = sum;
        }
    }
}

// ============ GPU朴素矩阵乘法 ============
/*
 * 朴素GPU实现：每个线程计算C矩阵的一个元素
 * 直接从全局内存读取A和B的数据
 * 
 * 性能瓶颈：
 * - 每个线程需要从全局内存读取width次数据
 * - 全局内存访问延迟高（约400个时钟周期）
 * - 内存带宽利用率低
 */
__global__ void matrixMulNaive(float* A, float* B, float* C, int width) {
    // 计算当前线程负责的元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // 计算点积
        for (int k = 0; k < width; k++) {
            // 每次迭代从全局内存读取数据
            sum += A[row * width + k] * B[k * width + col];
        }
        
        C[row * width + col] = sum;
    }
}

// ============ GPU共享内存优化矩阵乘法 ============
/*
 * 核心思想：将矩阵分块（Tiling）
 * 
 * 优化原理：
 * 1. 将大矩阵划分为小块（TILE_SIZE x TILE_SIZE）
 * 2. 每个线程块负责计算一个输出块
 * 3. 使用共享内存缓存输入矩阵块
 * 4. 线程协作加载数据，减少全局内存访问
 * 
 * 性能提升原因：
 * - 共享内存延迟约20个时钟周期（全局内存约400个周期）
 * - 数据重用：每个块被多个线程使用
 * - 减少全局内存访问次数：O(width) -> O(width/TILE_SIZE)
 */
__global__ void matrixMulShared(float* A, float* B, float* C, int width) {
    // ========== 1. 定义共享内存 ==========
    // 每个线程块共享这两个矩阵块
    // 注意：大小为 TILE_SIZE x TILE_SIZE
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // ========== 2. 计算线程索引 ==========
    // 当前线程在块内的位置
    int tx = threadIdx.x;  // 块内列索引
    int ty = threadIdx.y;  // 块内行索引
    
    // 当前线程在全局矩阵中的位置
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // 累加器
    float sum = 0.0f;
    
    // ========== 3. 分块计算 ==========
    // 遍历所有需要的块
    for (int m = 0; m < width / TILE_SIZE; ++m) {
        // ========== 4. 协作加载数据到共享内存 ==========
        // 每个线程加载一个元素
        // As[ty][tx]：从A矩阵加载，位置为 (row, m*TILE_SIZE + tx)
        // Bs[ty][tx]：从B矩阵加载，位置为 (m*TILE_SIZE + ty, col)
        
        int aCol = m * TILE_SIZE + tx;
        int bRow = m * TILE_SIZE + ty;
        
        // 从全局内存加载到共享内存
        As[ty][tx] = A[row * width + aCol];
        Bs[ty][tx] = B[bRow * width + col];
        
        // ========== 5. 同步：确保所有数据都加载完成 ==========
        // 重要：所有线程必须都执行这个同步
        // 否则某些线程可能读到未初始化的数据
        __syncthreads();
        
        // ========== 6. 计算当前块的贡献 ==========
        // 在共享内存中进行计算
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // ========== 7. 同步：确保计算完成 ==========
        // 在加载下一块数据前，确保所有线程都完成了当前块的计算
        __syncthreads();
    }
    
    // ========== 8. 写回结果 ==========
    C[row * width + col] = sum;
}

// ============ 辅助函数：初始化矩阵 ============
void initMatrix(float* mat, int width, float value = -1.0f) {
    if (value < 0) {
        // 随机初始化
        for (int i = 0; i < width * width; i++) {
            mat[i] = (float)(rand() % 100) / 100.0f;
        }
    } else {
        // 指定值初始化
        for (int i = 0; i < width * width; i++) {
            mat[i] = value;
        }
    }
}

// ============ 辅助函数：验证结果 ============
bool verifyResult(float* ref, float* result, int width, float epsilon = 1e-3) {
    for (int i = 0; i < width * width; i++) {
        if (fabs(ref[i] - result[i]) > epsilon) {
            printf("验证失败：位置 %d, 期望 %.6f, 实际 %.6f\n", 
                   i, ref[i], result[i]);
            return false;
        }
    }
    return true;
}

// ============ 辅助函数：使用CUDA Event计时 ============
float measureTime(cudaEvent_t start, cudaEvent_t stop) {
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// ============ 主函数 ============
int main() {
    // ========== 参数设置 ==========
    int width = 1024;  // 矩阵大小：1024 x 1024
    size_t size = width * width * sizeof(float);
    
    printf("=== 共享内存矩阵乘法性能对比 ===\n\n");
    printf("矩阵大小: %d x %d\n\n", width, width);
    
    // ========== 分配主机内存 ==========
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    float *h_C_naive = (float*)malloc(size);
    float *h_C_shared = (float*)malloc(size);
    
    // ========== 初始化数据 ==========
    srand(2024);  // 固定随机种子，保证可重复性
    initMatrix(h_A, width);
    initMatrix(h_B, width);
    
    // ========== 分配设备内存 ==========
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size));
    
    // ========== 拷贝数据到设备 ==========
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // ========== 创建CUDA Event用于计时 ==========
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // ========== 1. CPU计算 ==========
    printf("1. CPU计算中...\n");
    clock_t cpu_start = clock();
    matrixMulCPU(h_A, h_B, h_C_cpu, width);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("   CPU时间: %.2f ms\n\n", cpu_time);
    
    // ========== 2. GPU朴素实现 ==========
    printf("2. GPU朴素实现...\n");
    
    // 配置线程块和网格
    dim3 blockDim(16, 16);  // 每个线程块 16x16 = 256个线程
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (width + blockDim.y - 1) / blockDim.y);
    
    // 执行核函数并计时
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float naive_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&naive_time, start, stop));
    
    // 拷贝结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost));
    
    printf("   GPU朴素时间: %.2f ms\n", naive_time);
    printf("   加速比: %.2fx\n\n", cpu_time / naive_time);
    
    // ========== 3. GPU共享内存优化 ==========
    printf("3. GPU共享内存优化...\n");
    
    // 配置线程块和网格（使用TILE_SIZE）
    dim3 blockDimShared(TILE_SIZE, TILE_SIZE);
    dim3 gridDimShared(width / TILE_SIZE, width / TILE_SIZE);
    
    // 执行核函数并计时
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    matrixMulShared<<<gridDimShared, blockDimShared>>>(d_A, d_B, d_C, width);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float shared_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&shared_time, start, stop));
    
    // 拷贝结果回主机
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_shared, d_C, size, cudaMemcpyDeviceToHost));
    
    printf("   GPU共享内存时间: %.2f ms\n", shared_time);
    printf("   加速比: %.2fx (vs CPU)\n", cpu_time / shared_time);
    printf("   加速比: %.2fx (vs 朴素GPU)\n\n", naive_time / shared_time);
    
    // ========== 4. 验证结果 ==========
    printf("4. 验证结果...\n");
    bool naive_correct = verifyResult(h_C_cpu, h_C_naive, width);
    bool shared_correct = verifyResult(h_C_cpu, h_C_shared, width);
    
    if (naive_correct && shared_correct) {
        printf("   ✓ 所有结果正确！\n\n");
    } else {
        printf("   ✗ 结果验证失败！\n\n");
    }
    
    // ========== 5. 性能总结 ==========
    printf("=== 性能总结 ===\n");
    printf("CPU版本:          %8.2f ms\n", cpu_time);
    printf("GPU朴素版本:      %8.2f ms (加速 %.2fx)\n", 
           naive_time, cpu_time / naive_time);
    printf("GPU共享内存版本:  %8.2f ms (加速 %.2fx)\n", 
           shared_time, cpu_time / shared_time);
    printf("\n优化效果: 共享内存版本比朴素版本快 %.2fx\n", 
           naive_time / shared_time);
    
    // ========== 计算性能指标 ==========
    // GFLOPS: 每秒浮点运算次数
    // 矩阵乘法浮点运算次数 = 2 * width^3
    double flops = 2.0 * width * width * width;
    double gflops_naive = (flops / naive_time) / 1e6;
    double gflops_shared = (flops / shared_time) / 1e6;
    
    printf("\n=== 性能指标 ===\n");
    printf("GPU朴素版本: %.2f GFLOPS\n", gflops_naive);
    printf("GPU共享内存版本: %.2f GFLOPS\n", gflops_shared);
    
    // ========== 清理资源 ==========
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_naive);
    free(h_C_shared);
    
    printf("\n程序执行完成！\n");
    return 0;
}

/*
 * ========== 编译和运行 ==========
 * 
 * 编译命令:
 *   nvcc -O3 -arch=sm_80 shared_memory_matrix_mul.cu -o matrix_mul
 * 
 * 运行命令:
 *   ./matrix_mul
 * 
 * 注意事项:
 * 1. -arch=sm_80 适用于RTX 30系列GPU
 * 2. RTX 40系列使用 -arch=sm_89
 * 3. RTX 20系列使用 -arch=sm_75
 * 
 * ========== 性能分析 ==========
 * 
 * 使用nvprof分析:
 *   nvprof ./matrix_mul
 * 
 * 详细分析:
 *   nvprof --print-gpu-trace ./matrix_mul
 * 
 * ========== 预期输出 ==========
 * 
 * 矩阵大小: 1024 x 1024
 * 
 * CPU时间: ~1200 ms
 * GPU朴素时间: ~45 ms
 * GPU共享内存时间: ~18 ms
 * 
 * 加速比:
 * - GPU朴素 vs CPU: ~27x
 * - GPU共享内存 vs CPU: ~66x
 * - GPU共享内存 vs GPU朴素: ~2.5x
 */
