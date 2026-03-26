#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: PyTorch张量操作练习
=========================

本脚本演示PyTorch张量的基础操作,包括:
1. 张量创建的多种方式
2. 张量形状操作
3. CPU与GPU设备转移
4. 基本运算
5. 性能对比

适合初学者理解PyTorch的核心数据结构。
"""

import torch
import numpy as np
import time


def print_section(title):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def check_environment():
    """检查运行环境"""
    print_section("环境检查")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        
        # 显存信息
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU总显存: {total_memory:.2f} GB")
        
        device = torch.device('cuda')
        print(f"\n使用设备: GPU (cuda)")
    else:
        device = torch.device('cpu')
        print(f"\n使用设备: CPU")
        print("提示: 安装GPU版本的PyTorch以获得更好的性能")
    
    return device


def tensor_creation():
    """演示张量创建的多种方式"""
    print_section("1. 张量创建")
    
    # 1.1 从Python列表创建
    print("\n方法1: 从Python列表创建")
    x1 = torch.tensor([1, 2, 3, 4, 5])
    print(f"  一维张量: {x1}")
    print(f"  形状: {x1.shape}, 数据类型: {x1.dtype}")
    
    x2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"  二维张量:\n{x2}")
    print(f"  形状: {x2.shape}")
    
    # 1.2 使用工厂函数
    print("\n方法2: 使用工厂函数")
    
    # 全零张量
    zeros = torch.zeros(3, 4)
    print(f"  zeros(3, 4):\n{zeros}")
    
    # 全一张量
    ones = torch.ones(2, 3, 4)
    print(f"  ones(2, 3, 4) 形状: {ones.shape}")
    
    # 单位矩阵
    eye = torch.eye(5)
    print(f"  eye(5):\n{eye}")
    
    # 未初始化张量(速度快但不安全)
    empty = torch.empty(2, 3)
    print(f"  empty(2, 3) 形状: {empty.shape}")
    
    # 1.3 随机张量
    print("\n方法3: 创建随机张量")
    
    # 均匀分布 [0, 1)
    rand = torch.rand(3, 3)
    print(f"  rand(3, 3) - 均匀分布:\n{rand}")
    
    # 标准正态分布 N(0, 1)
    randn = torch.randn(3, 3)
    print(f"  randn(3, 3) - 标准正态分布:\n{randn}")
    
    # 整数随机 [low, high)
    randint = torch.randint(0, 10, (3, 3))
    print(f"  randint(0, 10, (3, 3)):\n{randint}")
    
    # 1.4 从NumPy创建
    print("\n方法4: 从NumPy数组创建")
    np_array = np.array([1, 2, 3, 4])
    tensor_from_np = torch.from_numpy(np_array)
    print(f"  NumPy数组: {np_array}")
    print(f"  转换后张量: {tensor_from_np}")
    print(f"  数据类型: {tensor_from_np.dtype} (注意:继承了NumPy的int32)")
    
    # 指定数据类型
    tensor_float = torch.from_numpy(np_array).float()
    print(f"  转换为float: {tensor_float}, dtype={tensor_float.dtype}")
    
    # 1.5 类似NumPy的函数
    print("\n方法5: 类似NumPy的函数")
    
    # arange
    arange = torch.arange(0, 10, 2)
    print(f"  arange(0, 10, 2): {arange}")
    
    # linspace
    linspace = torch.linspace(0, 1, 5)
    print(f"  linspace(0, 1, 5): {linspace}")
    
    # logspace
    logspace = torch.logspace(0, 2, 5)
    print(f"  logspace(0, 2, 5): {logspace}")


def tensor_properties():
    """演示张量属性和形状操作"""
    print_section("2. 张量属性和形状操作")
    
    # 创建一个3D张量
    x = torch.randn(2, 3, 4)
    print(f"原始张量形状: {x.shape}")
    print(f"维度数: {x.ndim}")
    print(f"元素总数: {x.numel()}")
    print(f"数据类型: {x.dtype}")
    print(f"设备: {x.device}")
    
    # 形状操作
    print("\n形状操作:")
    
    # reshape: 改变形状
    x_reshaped = x.reshape(6, 4)
    print(f"  reshape(6, 4): {x_reshaped.shape}")
    
    # view: 改变形状(共享内存)
    x_view = x.view(2, 12)
    print(f"  view(2, 12): {x_view.shape}")
    
    # flatten: 展平
    x_flat = x.flatten()
    print(f"  flatten(): {x_flat.shape}")
    
    # squeeze: 删除大小为1的维度
    x_sq = torch.randn(1, 3, 1, 4)
    print(f"\n  原始: {x_sq.shape}")
    x_squeezed = x_sq.squeeze()
    print(f"  squeeze(): {x_squeezed.shape}")
    
    # unsqueeze: 添加维度
    x_unsq = x.unsqueeze(0)
    print(f"  unsqueeze(0): {x_unsq.shape}")
    
    # transpose: 转置
    x_t = x.transpose(0, 1)
    print(f"  transpose(0, 1): {x_t.shape}")
    
    # permute: 重新排列维度
    x_perm = x.permute(2, 0, 1)
    print(f"  permute(2, 0, 1): {x_perm.shape}")


def device_transfer(device):
    """演示CPU与GPU之间的数据传输"""
    print_section("3. CPU与GPU设备转移")
    
    # 3.1 在CPU上创建张量
    print("\n在CPU上创建张量:")
    x_cpu = torch.randn(3, 3)
    print(f"  设备: {x_cpu.device}")
    
    # 3.2 移动到GPU
    print("\n移动到GPU的方法:")
    
    # 方法1: 使用.to()
    x_gpu = x_cpu.to(device)
    print(f"  方法1 - .to(device): {x_gpu.device}")
    
    # 方法2: 使用.cuda()
    if torch.cuda.is_available():
        x_gpu2 = x_cpu.cuda()
        print(f"  方法2 - .cuda(): {x_gpu2.device}")
        
        # 方法3: 指定GPU编号
        x_gpu3 = x_cpu.cuda(0)
        print(f"  方法3 - .cuda(0): {x_gpu3.device}")
    
    # 3.3 在GPU上直接创建
    print("\n直接在GPU上创建:")
    y_gpu = torch.randn(3, 3, device=device)
    print(f"  设备: {y_gpu.device}")
    
    # 3.4 从GPU移回CPU
    print("\n从GPU移回CPU:")
    y_cpu = y_gpu.cpu()
    print(f"  设备: {y_cpu.device}")
    
    # 3.5 注意事项
    print("\n⚠️ 重要注意事项:")
    print("  1. GPU上的张量只能与GPU上的张量运算")
    print("  2. CPU上的张量只能与CPU上的张量运算")
    print("  3. 混合运算会导致错误!")
    
    # 演示错误
    print("\n  ❌ 错误示例(已注释):")
    print("     # x_cpu = torch.randn(3, 3)")
    print("     # y_gpu = torch.randn(3, 3, device='cuda')")
    print("     # result = x_cpu + y_gpu  # RuntimeError!")
    
    # 正确做法
    print("\n  ✅ 正确做法:")
    if torch.cuda.is_available():
        y_cpu = y_gpu.cpu()
        result = x_cpu + y_cpu
        print(f"     先统一设备,再运算: {result.shape}")


def tensor_operations():
    """演示张量的基本运算"""
    print_section("4. 张量基本运算")
    
    # 创建示例张量
    x = torch.tensor([[1.0, 2.0, 3.0], 
                      [4.0, 5.0, 6.0]])
    y = torch.tensor([[1.0, 1.0, 1.0], 
                      [2.0, 2.0, 2.0]])
    
    print(f"x:\n{x}")
    print(f"y:\n{y}")
    
    # 4.1 算术运算
    print("\n算术运算:")
    print(f"  加法: x + y =\n{x + y}")
    print(f"  减法: x - y =\n{x - y}")
    print(f"  乘法(逐元素): x * y =\n{x * y}")
    print(f"  除法: x / y =\n{x / y}")
    print(f"  幂运算: x ** 2 =\n{x ** 2}")
    
    # 4.2 矩阵运算
    print("\n矩阵运算:")
    A = torch.randn(2, 3)
    B = torch.randn(3, 2)
    print(f"  A形状: {A.shape}, B形状: {B.shape}")
    
    # 矩阵乘法
    C = torch.matmul(A, B)
    print(f"  矩阵乘法 matmul(A, B) 形状: {C.shape}")
    
    # 或使用 @ 运算符
    C2 = A @ B
    print(f"  使用 @ 运算符: {C2.shape}")
    
    # 转置
    print(f"  转置 A.T 形状: {A.T.shape}")
    
    # 4.3 聚合运算
    print("\n聚合运算:")
    x = torch.randn(3, 4)
    print(f"x:\n{x}")
    print(f"  求和: {torch.sum(x)}")
    print(f"  均值: {torch.mean(x)}")
    print(f"  最大值: {torch.max(x)}")
    print(f"  最小值: {torch.min(x)}")
    print(f"  标准差: {torch.std(x)}")
    print(f"  方差: {torch.var(x)}")
    
    # 按维度聚合
    print("\n按维度聚合:")
    print(f"  按行求和(dim=1): {torch.sum(x, dim=1)}")
    print(f"  按列求和(dim=0): {torch.sum(x, dim=0)}")
    print(f"  按行最大值: {torch.max(x, dim=1)}")
    
    # 4.4 比较运算
    print("\n比较运算:")
    x = torch.tensor([1, 2, 3, 4, 5])
    print(f"x = {x}")
    print(f"  x > 3: {x > 3}")
    print(f"  x == 3: {x == 3}")
    print(f"  (x > 2) & (x < 4): {(x > 2) & (x < 4)}")
    
    # 4.5 索引和切片
    print("\n索引和切片:")
    x = torch.arange(12).reshape(3, 4)
    print(f"x:\n{x}")
    print(f"  第一行: x[0] = {x[0]}")
    print(f"  第一列: x[:, 0] = {x[:, 0]}")
    print(f"  右下角2x2: x[1:, 2:] =\n{x[1:, 2:]}")
    print(f"  条件索引(x > 5): {x[x > 5]}")


def performance_comparison(device):
    """对比CPU和GPU的性能差异"""
    print_section("5. CPU与GPU性能对比")
    
    # 测试不同规模
    sizes = [100, 500, 1000, 2000, 5000]
    
    print("\n矩阵乘法性能测试:")
    print(f"{'规模':<10} {'CPU时间':<12} {'GPU时间':<12} {'加速比':<10}")
    print("-" * 50)
    
    for size in sizes:
        # CPU版本
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start
        
        # GPU版本
        if torch.cuda.is_available():
            a_gpu = a_cpu.to(device)
            b_gpu = b_cpu.to(device)
            
            # 预热
            _ = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            
            # 计时
            start = time.time()
            c_gpu = torch.matmul(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            speedup = cpu_time / gpu_time
            print(f"{size:<10} {cpu_time:<12.4f} {gpu_time:<12.4f} {speedup:<10.2f}x")
        else:
            print(f"{size:<10} {cpu_time:<12.4f} {'N/A':<12} {'N/A':<10}")
    
    print("\n💡 观察:")
    print("  1. 数据规模越大,GPU加速越明显")
    print("  2. 小规模数据可能GPU反而更慢(传输开销)")
    print("  3. 对于大规模计算,GPU可以带来几十到几百倍加速")


def numpy_comparison():
    """对比PyTorch和NumPy"""
    print_section("6. PyTorch vs NumPy对比")
    
    size = 1000
    
    # NumPy
    print("\nNumPy性能:")
    a_np = np.random.randn(size, size)
    b_np = np.random.randn(size, size)
    
    start = time.time()
    c_np = np.matmul(a_np, b_np)
    numpy_time = time.time() - start
    print(f"  矩阵乘法({size}x{size}): {numpy_time:.4f}秒")
    
    # PyTorch CPU
    print("\nPyTorch CPU性能:")
    a_torch = torch.from_numpy(a_np)
    b_torch = torch.from_numpy(b_np)
    
    start = time.time()
    c_torch = torch.matmul(a_torch, b_torch)
    pytorch_cpu_time = time.time() - start
    print(f"  矩阵乘法({size}x{size}): {pytorch_cpu_time:.4f}秒")
    
    # PyTorch GPU
    if torch.cuda.is_available():
        print("\nPyTorch GPU性能:")
        device = torch.device('cuda')
        a_gpu = a_torch.to(device)
        b_gpu = b_torch.to(device)
        
        # 预热
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        pytorch_gpu_time = time.time() - start
        print(f"  矩阵乘法({size}x{size}): {pytorch_gpu_time:.4f}秒")
        
        print(f"\n加速比:")
        print(f"  PyTorch CPU vs NumPy: {numpy_time/pytorch_cpu_time:.2f}x")
        print(f"  PyTorch GPU vs NumPy: {numpy_time/pytorch_gpu_time:.2f}x")
        print(f"  PyTorch GPU vs CPU: {pytorch_cpu_time/pytorch_gpu_time:.2f}x")
    
    print("\n💡 总结:")
    print("  - PyTorch CPU与NumPy性能接近")
    print("  - PyTorch GPU提供显著加速")
    print("  - PyTorch API与NumPy高度兼容(90%+)")


def main():
    """主函数"""
    print("="*60)
    print("  PyTorch张量操作练习")
    print("  Day 3: Python GPU编程 - PyTorch入门")
    print("="*60)
    
    # 检查环境
    device = check_environment()
    
    # 演示各个主题
    tensor_creation()
    tensor_properties()
    device_transfer(device)
    tensor_operations()
    performance_comparison(device)
    numpy_comparison()
    
    # 总结
    print_section("学习总结")
    print("\n✅ 你已经学习了:")
    print("  1. 张量创建的多种方式")
    print("  2. 张量属性和形状操作")
    print("  3. CPU与GPU设备转移")
    print("  4. 张量基本运算")
    print("  5. CPU与GPU性能对比")
    print("  6. PyTorch与NumPy对比")
    
    print("\n🎯 关键要点:")
    print("  - 张量是PyTorch的核心数据结构")
    print("  - GPU可以提供几十到几百倍的性能提升")
    print("  - PyTorch API与NumPy高度兼容")
    print("  - 设备管理是GPU编程的关键")
    
    print("\n📚 下一步:")
    print("  - 运行 autograd_demo.py 学习自动微分")
    print("  - 运行 image_filter_gpu.py 学习GPU图像处理")
    print("  - 运行 performance_benchmark.py 进行性能测试")
    
    print("\n" + "="*60)
    print("练习完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
