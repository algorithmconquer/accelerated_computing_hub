# Day 4: 数学基础回顾与工具准备 - 项目代码

## 项目概述

本项目包含Day 4教程中所有实践代码,帮助初学者通过动手实践掌握GPU加速逆光刻技术所需的数学基础。

## 项目内容

### 1. 线性代数演示 (`linear_algebra_demo.py`)
- 矩阵创建与基本运算
- 特征值与特征向量计算
- 矩阵分解(SVD)
- 光刻掩模的矩阵表示

### 2. 傅里叶变换分析 (`fft_analysis.py`)
- 一维信号的FFT分析
- 二维图像的频谱分析
- 频域滤波实现
- 卷积定理验证

### 3. 优化算法演示 (`optimization_demo.py`)
- 梯度下降实现
- 不同优化器比较
- 优化路径可视化
- 学习率影响分析

### 4. 简化光刻仿真 (`lithography_simulation.py`)
- 掩模图案的矩阵表示
- 基于FFT的成像仿真
- 简单的掩模优化
- 结果可视化

### 5. 数值工具对比 (`numerical_tools_comparison.py`)
- NumPy vs PyTorch性能对比
- CPU vs GPU性能对比
- 不同规模的性能分析
- 工具选择建议

## 环境要求

### 硬件要求
- NVIDIA GPU (显存 ≥ 8GB)
- 支持 CUDA 11.0+

### 软件要求
- Python 3.8+
- PyTorch 2.0+ (支持CUDA)
- NumPy
- Matplotlib

## 安装步骤

```bash
# 1. 进入项目目录
cd code_2026_03_27

# 2. 安装依赖(如果尚未安装)
pip install torch torchvision numpy matplotlib

# 3. 验证GPU环境
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

## 运行步骤

### 步骤1:线性代数练习
```bash
python linear_algebra_demo.py
```
**预期输出**:
- 控制台显示矩阵运算结果
- 生成 `matrix_operations.png` (矩阵运算可视化)
- 生成 `eigenvalue_analysis.png` (特征值分析)

### 步骤2:傅里叶变换分析
```bash
python fft_analysis.py
```
**预期输出**:
- 控制台显示FFT分析结果
- 生成 `fft_1d_example.png` (一维FFT示例)
- 生成 `fft_2d_mask.png` (二维FFT示例)
- 生成 `convolution_theorem.png` (卷积定理验证)

### 步骤3:优化算法演示
```bash
python optimization_demo.py
```
**预期输出**:
- 控制台显示优化过程
- 生成 `optimization_example.png` (优化路径)
- 生成 `optimizer_comparison.png` (算法比较)

### 步骤4:简化光刻仿真
```bash
python lithography_simulation.py
```
**预期输出**:
- 控制台显示仿真过程
- 生成 `lithography_simulation_results.png` (仿真结果)
- 生成 `mask_optimization.png` (掩模优化过程)

### 步骤5:数值工具对比
```bash
python numerical_tools_comparison.py
```
**预期输出**:
- 控制台显示性能对比表格
- 生成 `performance_comparison.png` (性能对比图)

## 学习建议

### 对于初学者
1. **按顺序运行**: 先运行每个脚本,观察输出
2. **阅读代码**: 详细阅读代码注释,理解每一步
3. **修改参数**: 尝试修改参数,观察结果变化
4. **动手实践**: 完成课后练习,加深理解

### 学习时间分配
- 线性代数演示: 40分钟
- 傅里叶变换分析: 50分钟
- 优化算法演示: 50分钟
- 光刻仿真: 60分钟
- 数值工具对比: 30分钟
- **总计**: 约4-5小时

## 常见问题

### Q1: 提示CUDA不可用怎么办?
**A**: 检查以下几点:
1. 确认电脑有NVIDIA GPU
2. 安装了CUDA Toolkit
3. 安装了正确版本的PyTorch(GPU版本)

### Q2: 内存不足错误怎么办?
**A**: 尝试:
1. 减小矩阵/图像大小
2. 使用`torch.cuda.empty_cache()`清理显存
3. 降低批处理大小

### Q3: 如何验证结果正确性?
**A**: 参考 `expected_outputs.txt` 文件,里面有:
- 预期的控制台输出
- 预期的图像文件列表
- 关键结果的数值范围

### Q4: 为什么我的性能数据与教程不同?
**A**: 性能数据受以下因素影响:
- GPU型号(教程使用RTX 3090)
- GPU利用率
- 系统负载
- PyTorch版本

只要趋势一致(GPU显著快于CPU),结果就是合理的。

## 代码结构说明

每个Python文件都遵循以下结构:
```python
"""
文件说明
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def function_name():
    """
    函数说明
    """
    # 详细注释的代码
    pass

def main():
    """主函数"""
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 调用各个函数
    function_name()
    
    # 生成可视化
    plt.savefig('output.png')

if __name__ == "__main__":
    main()
```

## 扩展练习

完成基础练习后,可以尝试:
1. 实现其他矩阵分解方法(LU、QR)
2. 实现其他频域滤波器(带通、陷波)
3. 实现其他优化算法(RMSprop、Adagrad)
4. 优化光刻仿真模型

## 技术支持

如遇问题,请:
1. 查看教程文档 `2026_03_27.md`
2. 查看预期输出 `expected_outputs.txt`
3. 检查GPU环境是否正确配置
4. 尝试减少数据规模进行测试

## 下一步学习

完成本项目后,你将掌握:
- ✅ 线性代数在光刻中的应用
- ✅ 傅里叶变换的基本使用
- ✅ 优化算法的实现
- ✅ 数值计算工具的选择

准备好进入 **Day 5: CUDA编程基础深入** 了吗?
