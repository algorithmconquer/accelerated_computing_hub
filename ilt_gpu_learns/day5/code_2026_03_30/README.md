# Day 5 项目代码：CUDA核函数深入与性能优化

## 📋 项目概述

本项目包含Day 5学习的所有实践代码，涵盖CUDA核函数优化、共享内存使用、向量运算优化和归约算法对比。

**学习目标**:
- 掌握共享内存优化技巧
- 理解内存访问模式对性能的影响
- 学会使用性能分析工具
- 对比不同优化策略的性能差异

---

## 🗂️ 文件列表

### 1. 核心代码文件

| 文件名 | 描述 | 难度 |
|-------|------|-----|
| `shared_memory_matrix_mul.cu` | 共享内存矩阵乘法优化 | ⭐⭐⭐ |
| `vector_operations.cu` | GPU向量运算库 | ⭐⭐ |
| `reduction_comparison.cu` | 归约算法性能对比 | ⭐⭐⭐ |
| `performance_analysis.py` | 性能分析工具演示 | ⭐⭐ |

### 2. 辅助文件

| 文件名 | 描述 |
|-------|------|
| `README.md` | 本说明文档 |
| `expected_outputs.txt` | 预期输出示例 |

---

## 🚀 快速开始

### 环境要求

- CUDA Toolkit 11.0 或更高版本
- NVIDIA GPU（计算能力 ≥ 7.0）
- Python 3.8+（用于性能分析脚本）
- PyTorch 2.0+（用于性能对比）

### 编译所有CUDA程序

```bash
# 进入项目目录
cd code_2026_03_30

# 编译共享内存矩阵乘法
nvcc -O3 -arch=sm_80 shared_memory_matrix_mul.cu -o matrix_mul

# 编译向量运算库
nvcc -O3 -arch=sm_80 vector_operations.cu -o vector_ops

# 编译归约算法对比
nvcc -O3 -arch=sm_80 reduction_comparison.cu -o reduction
```

**注意**: `-arch=sm_80` 适用于Ampere架构（如RTX 30系列）。根据你的GPU调整：
- RTX 20系列: `-arch=sm_75`
- RTX 30系列: `-arch=sm_80`
- RTX 40系列: `-arch=sm_89`
- H100: `-arch=sm_90`

### 运行示例

```bash
# 1. 运行共享内存矩阵乘法
./matrix_mul

# 2. 运行向量运算库
./vector_ops

# 3. 运行归约算法对比
./reduction

# 4. 运行性能分析
python performance_analysis.py
```

---

## 📊 预期输出

### 1. 共享内存矩阵乘法

```
=== 共享内存矩阵乘法性能对比 ===

矩阵大小: 1024 x 1024

CPU计算时间: 1234.56 ms
GPU朴素实现时间: 45.23 ms
GPU共享内存优化时间: 18.67 ms

加速比:
  朴素GPU vs CPU: 27.30x
  共享内存GPU vs CPU: 66.13x
  共享内存GPU vs 朴素GPU: 2.42x

验证结果: 计算正确！
```

### 2. 向量运算库

```
=== GPU向量运算性能对比 ===

向量大小: 10000000 元素

1. 向量加法:
   CPU时间: 23.45 ms
   GPU时间: 0.82 ms
   加速比: 28.60x

2. 向量点积:
   CPU时间: 18.23 ms
   GPU归约时间: 0.45 ms
   加速比: 40.51x

3. 向量缩放:
   CPU时间: 15.67 ms
   GPU时间: 0.61 ms
   加速比: 25.69x

验证: 所有结果正确！
```

### 3. 归约算法对比

```
=== 归约算法性能对比 ===

向量大小: 16777216 元素 (16M)

方法1 - 朴素归约:
  时间: 2.34 ms
  带宽: 28.76 GB/s

方法2 - 共享内存归约:
  时间: 1.23 ms
  带宽: 54.73 GB/s
  加速比: 1.90x

方法3 - 循环展开归约:
  时间: 0.98 ms
  带宽: 68.69 GB/s
  加速比: 2.39x

验证: 所有方法结果正确！
```

---

## 🔍 代码详解

### 项目1: 共享内存矩阵乘法

**文件**: `shared_memory_matrix_mul.cu`

**核心思想**:
```
1. 将矩阵分块（Tile），每个块大小为 TILE_SIZE x TILE_SIZE
2. 每个线程块负责计算一个输出块
3. 使用共享内存缓存输入矩阵块
4. 所有线程协作加载共享内存，同步后计算
```

**性能提升原因**:
- 减少全局内存访问次数
- 共享内存延迟低（20周期 vs 400周期）
- 数据重用率高

**关键代码片段**:
```cuda
// 定义共享内存
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// 协作加载数据
As[ty][tx] = A[row * width + m * TILE_SIZE + tx];
Bs[ty][tx] = B[(m * TILE_SIZE + ty) * width + col];

// 同步确保加载完成
__syncthreads();

// 计算当前块的贡献
for (int k = 0; k < TILE_SIZE; ++k) {
    value += As[ty][k] * Bs[k][tx];
}
```

---

### 项目2: 向量运算库

**文件**: `vector_operations.cu`

**包含功能**:

1. **向量加法**（向量化优化）
   - 使用 `float4` 向量化访问
   - 一次处理4个元素

2. **向量点积**（归约优化）
   - 树形归约算法
   - 使用共享内存

3. **向量缩放**（带宽优化）
   - 合并内存访问
   - 最大带宽利用率

**性能对比**:
- CPU实现：简单的for循环
- GPU实现：优化后的CUDA核函数

---

### 项目3: 归约算法对比

**文件**: `reduction_comparison.cu`

**三种方法对比**:

1. **朴素归约**
   - 简单的原子操作
   - 性能最差
   - 理解基准

2. **共享内存归约**
   - 树形归约结构
   - 使用共享内存
   - 性能中等

3. **循环展开归约**
   - 展开循环减少开销
   - Warp级优化
   - 性能最优

**学习要点**:
- 理解归约算法的演进
- 分析每一步优化的收益
- 掌握性能分析技巧

---

### 项目4: 性能分析工具

**文件**: `performance_analysis.py`

**功能**:
1. 使用 `nvprof` 分析CUDA程序
2. 使用 `cudaEvent` 精确计时
3. 计算内存带宽利用率
4. 分析占用率

**使用方法**:
```bash
# 基础性能分析
python performance_analysis.py

# 详细性能报告
nvprof --print-gpu-trace python performance_analysis.py

# 分析特定指标
nvprof --metrics gld_efficiency,gst_efficiency python performance_analysis.py
```

---

## 💡 优化技巧总结

### 1. 内存优化

| 技巧 | 性能提升 | 代码示例 |
|-----|---------|---------|
| 合并访问 | 10-100x | 连续线程访问连续地址 |
| 共享内存 | 2-10x | `__shared__ float tile[32][32]` |
| 向量化 | 2-4x | `float4 val = data[idx]` |
| 内存对齐 | 1.5-2x | `__align__(16)` |

### 2. 计算优化

| 技巧 | 性能提升 | 代码示例 |
|-----|---------|---------|
| 循环展开 | 1.5-2x | `#pragma unroll 4` |
| 避免分化 | 2-3x | 使用条件表达式 |
| 减少寄存器 | 1.2-1.5x | 重用变量 |

### 3. 同步优化

| 技巧 | 注意事项 |
|-----|---------|
| `__syncthreads()` | 块内所有线程必须执行 |
| 原子操作 | 确保线程安全 |
| Warp级同步 | 使用 `__shfl_xor_sync` |

---

## 🧪 性能测试

### 测试脚本

```bash
# 完整性能测试
#!/bin/bash

echo "=== Day 5 性能测试 ==="

# 编译所有程序
echo "编译程序..."
nvcc -O3 -arch=sm_80 shared_memory_matrix_mul.cu -o matrix_mul
nvcc -O3 -arch=sm_80 vector_operations.cu -o vector_ops
nvcc -O3 -arch=sm_80 reduction_comparison.cu -o reduction

# 运行测试
echo -e "\n1. 矩阵乘法性能测试"
./matrix_mul

echo -e "\n2. 向量运算性能测试"
./vector_ops

echo -e "\n3. 归约算法性能测试"
./reduction

echo -e "\n4. 性能分析"
python performance_analysis.py

echo -e "\n=== 测试完成 ==="
```

### 性能基准

**预期性能**（RTX 3090）:

| 算法 | 性能 | 带宽利用率 |
|-----|------|-----------|
| 共享内存矩阵乘法 | 15-20 ms (1024x1024) | 70-80% |
| 向量加法 | 0.5-1.0 ms (10M元素) | 80-90% |
| 归约求和 | 0.8-1.2 ms (16M元素) | 60-70% |

---

## 🐛 常见问题

### 问题1: 编译错误 "unsupported GNU version"

**原因**: GCC版本过高

**解决**:
```bash
# 使用较低版本的GCC
nvcc -ccbin g++-9 -O3 -arch=sm_80 program.cu -o program
```

### 问题2: 运行时错误 "invalid device ordinal"

**原因**: GPU架构设置错误

**解决**:
```bash
# 查询你的GPU计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 根据结果设置 -arch 参数
# 例如: 计算能力 8.6 → -arch=sm_86
```

### 问题3: 性能低于预期

**排查步骤**:
1. 检查是否使用优化的编译选项 `-O3`
2. 检查GPU架构是否正确设置
3. 使用 `nvidia-smi` 查看GPU是否在其他任务中
4. 使用 `nvprof` 分析瓶颈

### 问题4: 共享内存不足

**现象**: "too much shared memory"

**解决**:
- 减小 TILE_SIZE
- 检查共享内存使用量：`sizeof(float) * TILE_SIZE * TILE_SIZE`
- 查询GPU共享内存限制：`cudaDeviceProp.sharedMemPerBlock`

---

## 📚 学习建议

### 学习顺序

1. **先运行**：直接运行所有程序，看输出结果
2. **再读代码**：详细阅读每个CUDA核函数
3. **理解优化**：对比不同优化策略的性能差异
4. **修改实验**：尝试修改参数，观察性能变化
5. **性能分析**：使用 `nvprof` 分析瓶颈

### 实验建议

1. **调整矩阵大小**：观察不同大小的性能变化
2. **调整线程块大小**：尝试 128, 256, 512
3. **调整TILE_SIZE**：尝试 16, 32, 64
4. **对比CPU性能**：理解GPU加速的价值

### 思考题

1. 为什么共享内存矩阵乘法比朴素实现快2-3倍？
2. 向量化访问为什么能提高内存带宽利用率？
3. 归约算法中，为什么循环展开能提升性能？
4. 如何选择最优的线程块大小？

---

## 🔗 相关资源

### NVIDIA文档

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Profiler User Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)

### 学习资源

- Day 2教程：CUDA线程组织
- Day 4教程：数学基础回顾
- NVIDIA GTC教程：GPU性能优化

---

## ✅ 检查清单

完成以下任务后，你可以进入Day 6的学习：

- [ ] 成功编译所有CUDA程序
- [ ] 运行并理解共享内存矩阵乘法
- [ ] 运行并理解向量运算优化
- [ ] 运行并理解归约算法对比
- [ ] 使用性能分析工具分析至少一个程序
- [ ] 记录性能数据，理解优化效果

---

**准备好了吗？让我们开始今天的实践！** 🚀
