# Day 6: CUDA流与并发执行 - 项目代码

## 项目概述

本项目包含Day 6教程的所有实践代码，涵盖CUDA流、事件、流水线并行等核心概念。

## 项目结构

```
code_2026_03_31/
├── README.md                        # 本文件
├── stream_comparison.cu             # 单流与多流性能对比
├── pipeline_vector_add.cu           # 流水线并行向量加法
├── event_synchronization.cu         # 事件同步示例
├── pinned_memory_demo.cu            # 固定内存演示
├── stream_utils.h                   # 流工具函数头文件
├── stream_benchmark.py              # 流性能基准测试（Python版）
└── expected_outputs.md              # 预期输出示例
```

## 环境要求

- NVIDIA GPU (计算能力 ≥ 7.0)
- CUDA Toolkit (≥ 11.0)
- GCC/G++ (支持C++11)
- Python 3.8+ (用于Python版本)
- PyTorch 2.0+ (用于Python版本)

## 快速开始

### 1. 编译所有CUDA程序

```bash
# 进入项目目录
cd code_2026_03_31

# 编译所有CUDA程序
nvcc -O3 -arch=sm_80 stream_comparison.cu -o stream_comparison
nvcc -O3 -arch=sm_80 pipeline_vector_add.cu -o pipeline_vector_add
nvcc -O3 -arch=sm_80 event_synchronization.cu -o event_synchronization
nvcc -O3 -arch=sm_80 pinned_memory_demo.cu -o pinned_memory_demo
```

**注意**: 将 `-arch=sm_80` 替换为你的GPU架构：
- RTX 30系列: sm_86
- RTX 40系列: sm_89
- A100: sm_80
- H100: sm_90

### 2. 运行示例

```bash
# 1. 单流与多流对比
./stream_comparison

# 2. 流水线并行向量加法
./pipeline_vector_add

# 3. 事件同步示例
./event_synchronization

# 4. 固定内存演示
./pinned_memory_demo

# 5. Python版本基准测试
python stream_benchmark.py
```

## 项目详解

### 1. stream_comparison.cu - 单流与多流性能对比

**功能**:
- 实现单流执行版本
- 实现2、4、8流并发版本
- 对比性能差异
- 输出加速比

**核心概念**:
- CUDA流创建与销毁
- 异步内存传输
- 流同步
- 性能测量

**运行示例**:
```
=== CUDA流性能对比测试 ===
数据大小: 64 MB
迭代次数: 10

单流版本:
  平均时间: 45.2 ms
  
2流版本:
  平均时间: 32.8 ms
  加速比: 1.38x
  
4流版本:
  平均时间: 27.3 ms
  加速比: 1.66x
  
8流版本:
  平均时间: 26.1 ms
  加速比: 1.73x

最佳配置: 4个流
```

### 2. pipeline_vector_add.cu - 流水线并行向量加法

**功能**:
- 实现流水线式的向量加法
- H2D传输、计算、D2H传输并行
- 使用固定内存
- 可视化时间线

**核心概念**:
- 流水线并行
- 固定内存分配
- 多流协调
- 传输-计算重叠

**运行示例**:
```
=== 流水线并行向量加法 ===
数据总量: 256 MB
分块数: 4
每块大小: 64 MB

时间线分析:
流0: [H2D: 8.2ms][Compute: 12.5ms][D2H: 8.1ms]
流1:    [H2D: 8.1ms][Compute: 12.4ms][D2H: 8.0ms]
流2:       [H2D: 8.3ms][Compute: 12.6ms][D2H: 8.2ms]
流3:          [H2D: 8.2ms][Compute: 12.5ms][D2H: 8.1ms]

总时间: 37.3 ms
顺序执行时间: 116.0 ms
加速比: 3.11x

传输-计算重叠率: 67.3%
```

### 3. event_synchronization.cu - 事件同步示例

**功能**:
- 使用事件进行流同步
- 精确GPU计时
- 流间依赖管理
- 多阶段流程控制

**核心概念**:
- CUDA事件创建与销毁
- 事件记录与同步
- 流等待事件
- 时间测量

**运行示例**:
```
=== CUDA事件同步演示 ===

阶段1: 数据预处理 (流0)
  执行时间: 5.23 ms

阶段2: 主计算 (流1, 等待流0)
  流1等待流0事件...
  执行时间: 15.67 ms

阶段3: 后处理 (流2, 等待流1)
  流2等待流1事件...
  执行时间: 3.12 ms

总执行时间: 24.02 ms

事件同步正确: ✓
```

### 4. pinned_memory_demo.cu - 固定内存演示

**功能**:
- 对比固定内存与可分页内存
- 测量传输性能差异
- 演示异步传输
- 内存使用分析

**核心概念**:
- cudaMallocHost vs malloc
- cudaMemcpyAsync
- DMA传输
- 内存锁页

**运行示例**:
```
=== 固定内存演示 ===
数据大小: 128 MB

可分页内存 (malloc):
  同步传输时间: 12.5 ms
  带宽: 10.2 GB/s

固定内存 (cudaMallocHost):
  异步传输时间: 8.3 ms
  带宽: 15.4 GB/s
  
性能提升: 1.51x

内存分析:
  固定内存占用: 128 MB
  系统总内存: 32 GB
  占比: 0.4%
```

### 5. stream_benchmark.py - Python性能基准测试

**功能**:
- 使用PyTorch实现流并发
- 性能基准测试
- 可视化性能曲线
- 自动寻找最佳流数量

**核心概念**:
- torch.cuda.Stream
- torch.cuda.Event
- 性能分析与可视化

**运行示例**:
```bash
python stream_benchmark.py
```

输出:
```
=== PyTorch CUDA流基准测试 ===
GPU: NVIDIA GeForce RTX 3080
CUDA版本: 11.8

测试配置:
  数据大小: 64 MB
  流数量: [1, 2, 4, 8, 16]

性能结果:
  1个流: 42.3 ms (基准)
  2个流: 30.5 ms (1.39x)
  4个流: 25.7 ms (1.65x)
  8个流: 24.8 ms (1.71x)
  16个流: 25.1 ms (1.68x)

最佳流数量: 8
建议配置: 4-8个流
```

## 关键代码片段

### 创建和使用CUDA流

```cuda
// 创建流
cudaStream_t stream;
cudaStreamCreate(&stream);

// 在流中执行操作
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_data);
cudaMemcpyAsync(h_result, d_data, size, cudaMemcpyDeviceToHost, stream);

// 同步
cudaStreamSynchronize(stream);

// 销毁流
cudaStreamDestroy(stream);
```

### 使用CUDA事件计时

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... 执行操作 ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("时间: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### 流水线并行模式

```cuda
const int N_STREAMS = 4;
cudaStream_t streams[N_STREAMS];

for (int i = 0; i < N_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
}

for (int i = 0; i < N_STREAMS; i++) {
    int offset = i * chunk_size;
    
    cudaMemcpyAsync(d_data + offset, h_data + offset, 
                    chunk_size, cudaMemcpyHostToDevice, streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(d_data + offset);
    cudaMemcpyAsync(h_result + offset, d_data + offset, 
                    chunk_size, cudaMemcpyDeviceToHost, streams[i]);
}

for (int i = 0; i < N_STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
}
```

## 性能优化建议

### 1. 流数量选择
- 小数据量 (< 1MB): 1-2个流
- 中等数据量 (1-64MB): 2-4个流
- 大数据量 (> 64MB): 4-8个流

### 2. 固定内存使用
- 频繁传输的数据使用固定内存
- 不要过度使用（影响系统性能）
- 大块传输收益更明显

### 3. 传输-计算比例
- 计算密集型: 流水线效果显著
- 传输密集型: 需要优化数据布局
- 均衡型: 最适合多流并发

### 4. 流同步策略
- 批量同步: `cudaDeviceSynchronize()`
- 单流同步: `cudaStreamSynchronize()`
- 事件同步: `cudaEventSynchronize()`

## 常见问题排查

### Q1: 程序崩溃或卡死
- 检查是否在同步前访问数据
- 确认固定内存是否正确释放
- 验证流数量是否合理

### Q2: 性能没有提升
- 增大数据量
- 检查是否使用了固定内存
- 确认硬件是否支持并发

### Q3: 编译错误
- 检查GPU架构设置
- 确认CUDA版本兼容性
- 查看编译器错误信息

## 学习建议

1. **循序渐进**: 先运行单个示例，理解原理
2. **修改参数**: 尝试不同的数据大小和流数量
3. **对比分析**: 观察单流和多流的性能差异
4. **实践应用**: 将技术应用到自己的项目中

## 扩展练习

1. **练习1**: 实现一个使用8个流的矩阵乘法
2. **练习2**: 对比不同数据大小的最佳流数量
3. **练习3**: 实现动态负载均衡的流调度
4. **练习4**: 使用Nsight Systems分析流并发

## 参考资源

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)

---

**作者**: GPU ILT学习体系
**日期**: 2026-03-31
**版本**: 1.0
