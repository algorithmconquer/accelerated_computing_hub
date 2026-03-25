# Day 2 项目：CUDA编程基础深入

## 项目概述

本项目的目标是深入理解CUDA编程模型，通过实践掌握GPU性能优化技巧。我们将实现三个核心示例：

1. **线程组织可视化** - 理解CUDA线程层次结构
2. **共享内存示例** - 掌握GPU内存优化技术
3. **矩阵乘法优化** - 综合运用所学知识进行性能优化

## 项目文件结构

```
code_2026_03_25/
├── thread_organization.cu       # 线程组织可视化示例
├── shared_memory_demo.cu        # 共享内存使用示例
├── matrix_multiplication.cu     # 矩阵乘法优化示例
├── verify_gpu_performance.py    # GPU性能验证脚本
├── README.md                    # 本文档
└── expected_outputs.txt         # 预期输出示例
```

## 环境要求

### 硬件要求
- NVIDIA GPU (计算能力 ≥ 6.0)
- 显存 ≥ 8GB
- 内存 ≥ 16GB

### 软件要求
- CUDA Toolkit 11.0 或更高版本
- Python 3.8+ (用于PyTorch验证脚本)
- PyTorch 2.0+ (支持CUDA)
- GCC/G++ 编译器

## 编译与运行

### 1. 编译CUDA程序

```bash
# 编译所有CUDA程序
nvcc -o thread_organization thread_organization.cu
nvcc -o shared_memory_demo shared_memory_demo.cu
nvcc -o matrix_multiplication matrix_multiplication.cu

# 或者使用优化选项编译
nvcc -O3 -o thread_organization thread_organization.cu
nvcc -O3 -o shared_memory_demo shared_memory_demo.cu
nvcc -O3 -o matrix_multiplication matrix_multiplication.cu
```

### 2. 运行CUDA程序

```bash
# 运行线程组织示例
./thread_organization

# 运行共享内存示例
./shared_memory_demo

# 运行矩阵乘法性能测试
./matrix_multiplication
```

### 3. 运行Python验证脚本

```bash
# 安装依赖
pip install torch numpy

# 运行GPU性能验证
python verify_gpu_performance.py
```

## 详细说明

### 1. thread_organization.cu - 线程组织可视化

**目标**：理解CUDA线程层次结构

**学习要点**：
- Grid和Block的维度设置
- 一维和二维网格的使用
- 线程索引计算方法
- 边界检查的重要性

**预期输出**：
```
=== 测试一维网格 ===
数据大小: 1000
块大小: 256
块数量: 4
总线程数: 1024

前10个元素:
  data[0] = 0.00
  data[1] = 2.00
  data[2] = 4.00
  ...

=== 测试二维网格 ===
图像大小: 640 x 480 = 307200 像素
块大小: 16 x 16 = 256 线程/块
网格大小: 40 x 30 = 1200 块
总线程数: 307200
```

**关键代码解读**：

```cpp
// 一维索引计算
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 二维索引计算
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;  // 行优先存储

// 边界检查（非常重要！）
if (idx < n) {
    // 处理数据
}
```

### 2. shared_memory_demo.cu - 共享内存示例

**目标**：掌握共享内存的使用和性能优化

**学习要点**：
- 共享内存的声明和分配
- __syncthreads()同步机制
- Bank冲突的避免
- 并行归约算法

**预期输出**：
```
=== 归约求和性能测试 ===
全局内存版本:
  时间: 1.234 ms
  结果: 1048576.00 (期望: 1048576.00)

共享内存版本:
  时间: 0.567 ms
  结果: 1048576.00 (期望: 1048576.00)

=== 矩阵转置测试 ===
验证转置结果...
✓ 转置正确!
```

**性能对比分析**：
- 全局内存版本：多次访问全局内存，延迟高
- 共享内存版本：数据复用，减少全局内存访问

**关键代码解读**：

```cpp
// 动态共享内存
extern __shared__ float sdata[];

// 静态共享内存
__shared__ float tile[16][16];

// 并行归约
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();  // 必须同步！
}
```

### 3. matrix_multiplication.cu - 矩阵乘法优化

**目标**：综合运用线程组织和内存优化技术

**学习要点**：
- CPU vs GPU性能对比
- 全局内存访问模式
- 分块（Tiling）技术
- 性能分析和GFLOPS计算

**预期输出**（RTX 3090）：
```
矩阵乘法性能测试
矩阵尺寸: A(1024x1024) × B(1024x1024) = C(1024x1024)

[1] CPU版本
  时间: 5234.567 ms

[2] GPU朴素版本
  时间: 48.123 ms
  加速比: 108.77x
  验证: ✓

[3] GPU共享内存版本
  时间: 4.567 ms
  加速比: 1146.23x (vs CPU)
  验证: ✓

性能分析
总浮点运算: 2147.48 GFLOPS

CPU版本:
  性能: 0.41 GFLOPS

GPU朴素版本:
  性能: 44.63 GFLOPS

GPU共享内存版本:
  性能: 470.28 GFLOPS
```

**分块技术原理**：

```
将大矩阵分成小块（如32x32）：
┌─────┬─────┬─────┐
│ As₀₀│ As₀₁│ ... │  A的小块
├─────┼─────┼─────┤
│ ... │ ... │ ... │
└─────┴─────┴─────┘

优势：
1. 减少全局内存访问次数
2. 利用共享内存的快速访问
3. 提高数据局部性
```

**关键代码解读**：

```cpp
// 加载数据到共享内存
As[threadIdx.y][threadIdx.x] = A[aRow * k + aCol];
Bs[threadIdx.y][threadIdx.x] = B[bRow * n + bCol];
__syncthreads();

// 计算部分积
for (int l = 0; l < TILE_SIZE; l++) {
    sum += As[threadIdx.y][l] * Bs[l][threadIdx.x];
}
__syncthreads();
```

### 4. verify_gpu_performance.py - GPU性能验证

**目标**：验证GPU的计算能力和内存带宽

**测试内容**：
- GPU设备信息查询
- 矩阵乘法性能测试
- 内存带宽测试（H2D/D2H）
- GPU内部内存带宽测试
- 共享内存加速效果演示

**运行结果示例**：
```
GPU设备信息
GPU名称: NVIDIA GeForce RTX 3090
CUDA版本: 11.8
PyTorch版本: 2.0.1
计算能力: 8.6
显存总量: 24.00 GB

矩阵乘法性能测试:
矩阵大小         时间(ms)         GFLOPS          
------------------------------------------------------------
512x512         0.123           4374.50        
1024x1024       0.456           4730.23        
2048x2048       2.345           7340.56        
4096x4096       15.678          8775.89        

内存拷贝性能测试 (Host <-> Device):
大小(MB)        H2D带宽(GB/s)     D2H带宽(GB/s)     
------------------------------------------------------------
1               5.23            4.89            
10              8.45            7.92            
100             11.23           10.87           
500             12.34           11.56           
1000            12.67           11.89           
```

## 性能优化技巧总结

### 1. 线程组织优化
- 选择合适的Block大小（通常256或512）
- 确保足够的占用率（>50%）
- 避免分支分歧

### 2. 内存访问优化
- 使用共享内存减少全局内存访问
- 确保合并访问（Coalesced Access）
- 避免Bank冲突

### 3. 分块技术
- 将大问题分解为小块
- 数据复用，减少重复加载
- 平衡计算与内存访问

### 4. 性能分析方法
- 使用CUDA Event精确计时
- 计算GFLOPS和带宽利用率
- 对比CPU性能建立基准

## 常见问题排查

### 问题1：编译错误 - 找不到CUDA头文件
```bash
# 检查CUDA安装
nvcc --version

# 设置环境变量
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 问题2：运行时错误 - CUDA初始化失败
```bash
# 检查GPU驱动
nvidia-smi

# 检查CUDA设备
python -c "import torch; print(torch.cuda.is_available())"
```

### 问题3：结果不正确
- 检查边界条件
- 验证同步操作
- 检查索引计算

### 问题4：性能不佳
- 检查占用率
- 分析内存访问模式
- 使用nvprof或nsight工具分析

## 进一步学习

### 推荐资源
1. **CUDA C++ Programming Guide** - NVIDIA官方文档
2. **CUDA C++ Best Practices Guide** - 性能优化指南
3. **GPU Gems 3** - GPU编程实战案例

### 进阶练习
1. 实现不同尺寸的分块矩阵乘法
2. 优化归约算法（如两两归约、树形归约）
3. 实现卷积运算的GPU优化
4. 使用Tensor Core加速矩阵运算

## 预期输出示例

完整的预期输出示例请查看 `expected_outputs.txt` 文件。

## 总结

通过本项目的学习，您应该已经掌握了：

✅ CUDA线程组织和执行模型
✅ GPU内存层次结构和优化策略
✅ 共享内存的使用方法和性能优势
✅ 矩阵乘法的GPU优化实现
✅ 性能分析和基准测试方法

这些知识将为后续学习光刻仿真和ILT算法实现奠定坚实的基础。
