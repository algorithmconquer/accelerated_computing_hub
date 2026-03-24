# Day 1: GPU加速逆光刻技术入门项目

## 项目简介

本项目是GPU加速逆光刻技术(ILT)学习系列的第一天实践项目,包含:
- 环境检查脚本
- CUDA向量加法示例
- 第一个GPU加速ILT演示程序

## 项目结构

```
code_2026_03_24/
├── README.md                    # 本文件
├── check_environment.py         # GPU环境检查脚本
├── vector_add.cu               # CUDA向量加法示例
├── first_ilt_demo.py           # ILT演示程序
└── results/                    # 输出结果目录
```

## 环境要求

### 硬件要求
- NVIDIA GPU: GTX 1060或更高
- 显存: ≥6GB
- 内存: ≥16GB

### 软件要求
- CUDA Toolkit 12.1
- Python 3.10+
- PyTorch 2.0+
- cuFFT库(随CUDA安装)

## 安装步骤

### 1. 创建Python虚拟环境

```bash
conda create -n ilt_gpu python=3.10
conda activate ilt_gpu
```

### 2. 安装PyTorch(支持CUDA)

```bash
# 根据你的CUDA版本选择
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. 安装其他依赖

```bash
pip install matplotlib numpy
```

### 4. 验证环境

```bash
python check_environment.py
```

## 使用说明

### 运行CUDA向量加法示例

```bash
# 编译
nvcc -o vector_add vector_add.cu

# 运行
./vector_add
```

### 运行GPU环境检查

```bash
python check_environment.py
```

### 运行第一个ILT演示程序

```bash
python first_ilt_demo.py
```

**预期输出**:
- 控制台输出优化过程
- 生成可视化图像: `results/first_ilt_result.png`

## 代码说明

### check_environment.py

检查GPU环境的完整脚本,验证:
- PyTorch CUDA支持
- GPU设备信息
- 显存情况
- 简单计算测试

### vector_add.cu

经典的CUDA入门示例:
- 核函数定义
- 线程索引计算
- 内存分配与传输
- 核函数启动

### first_ilt_demo.py

简化版ILT演示程序:
- 目标图案生成
- 光刻仿真(高斯模糊近似)
- 梯度下降优化
- 结果可视化

## 学习目标

完成本项目后,您应该能够:
1. ✅ 理解CUDA程序的基本结构
2. ✅ 掌握PyTorch GPU编程基础
3. ✅ 理解ILT优化的基本流程
4. ✅ 能够运行和修改示例代码

## 常见问题

### Q1: CUDA不可用怎么办?

**解决方案**:
1. 检查GPU驱动是否安装: `nvidia-smi`
2. 确认CUDA版本与PyTorch版本匹配
3. 重新安装PyTorch: `pip install --force-reinstall torch torchvision`

### Q2: 显存不足怎么办?

**解决方案**:
1. 减小图案尺寸: `create_target_pattern(128)`
2. 减小批处理大小
3. 使用混合精度训练: `with torch.cuda.amp.autocast():`

### Q3: 优化不收敛怎么办?

**解决方案**:
1. 调整学习率: `optimize(lr=0.1)`
2. 增加迭代次数: `optimize(iterations=200)`
3. 尝试不同优化器: `torch.optim.SGD`

## 下一步学习

完成今天的项目后,建议:
1. 阅读CUDA编程指南
2. 理解GPU内存层次
3. 学习并行算法设计

明天我们将学习:
- CUDA线程组织与内存管理
- 性能优化技巧
- 矩阵乘法优化实现

## 参考资料

- [NVIDIA CUDA文档](https://docs.nvidia.com/cuda/)
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [OpenILT项目](https://github.com/OpenOPC/OpenILT)

## 许可证

本项目仅供学习使用。
