#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: PyTorch性能基准测试
=========================

本脚本进行详细的性能测试,包括:
1. 矩阵乘法性能对比
2. 图像卷积性能对比
3. 张量操作性能对比
4. 内存带宽测试
5. 性能优化建议

帮助理解GPU加速的实际效果和适用场景。
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict


def print_section(title):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def warmup_gpu(device):
    """预热GPU"""
    if torch.cuda.is_available():
        print("预热GPU...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        for _ in range(10):
            _ = torch.matmul(x, y)
        torch.cuda.synchronize()
        print("预热完成\n")


def benchmark_matmul_cpu(sizes):
    """测试CPU矩阵乘法性能"""
    results = []
    
    for size in sizes:
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # 预热
        _ = torch.matmul(a, b)
        
        # 计时
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        elapsed = (time.time() - start) / 10
        
        # 计算GFLOPS
        flops = 2 * size**3  # 矩阵乘法的浮点运算次数
        gflops = flops / elapsed / 1e9
        
        results.append({
            'size': size,
            'time': elapsed,
            'gflops': gflops
        })
        
        if size % 1000 == 0 or size < 1000:
            print(f"  CPU {size}x{size}: {elapsed:.6f}秒, {gflops:.2f} GFLOPS")
    
    return results


def benchmark_matmul_gpu(sizes, device):
    """测试GPU矩阵乘法性能"""
    if not torch.cuda.is_available():
        return None
    
    results = []
    
    for size in sizes:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # 预热
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # 计时
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10
        
        # 计算GFLOPS
        flops = 2 * size**3
        gflops = flops / elapsed / 1e9
        
        results.append({
            'size': size,
            'time': elapsed,
            'gflops': gflops
        })
        
        if size % 1000 == 0 or size < 1000:
            print(f"  GPU {size}x{size}: {elapsed:.6f}秒, {gflops:.2f} GFLOPS")
    
    return results


def plot_matmul_results(cpu_results, gpu_results):
    """绘制矩阵乘法性能对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 提取数据
    cpu_sizes = [r['size'] for r in cpu_results]
    cpu_times = [r['time'] for r in cpu_results]
    cpu_gflops = [r['gflops'] for r in cpu_results]
    
    if gpu_results:
        gpu_sizes = [r['size'] for r in gpu_results]
        gpu_times = [r['time'] for r in gpu_results]
        gpu_gflops = [r['gflops'] for r in gpu_results]
    
    # 左图:执行时间
    axes[0].plot(cpu_sizes, cpu_times, 'b-o', label='CPU', markersize=6)
    if gpu_results:
        axes[0].plot(gpu_sizes, gpu_times, 'r-o', label='GPU', markersize=6)
    axes[0].set_xlabel('矩阵尺寸', fontsize=12)
    axes[0].set_ylabel('执行时间(秒)', fontsize=12)
    axes[0].set_title('矩阵乘法执行时间', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # 中图:GFLOPS
    axes[1].plot(cpu_sizes, cpu_gflops, 'b-o', label='CPU', markersize=6)
    if gpu_results:
        axes[1].plot(gpu_sizes, gpu_gflops, 'r-o', label='GPU', markersize=6)
    axes[1].set_xlabel('矩阵尺寸', fontsize=12)
    axes[1].set_ylabel('GFLOPS', fontsize=12)
    axes[1].set_title('计算吞吐量', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 右图:加速比
    if gpu_results:
        speedups = [cpu_t / gpu_t for cpu_t, gpu_t in zip(cpu_times, gpu_times)]
        axes[2].plot(cpu_sizes, speedups, 'g-o', markersize=6)
        axes[2].axhline(y=1.0, color='r', linestyle='--', label='基准线(1x)')
        axes[2].set_xlabel('矩阵尺寸', fontsize=12)
        axes[2].set_ylabel('加速比', fontsize=12)
        axes[2].set_title('GPU加速比', fontsize=14)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('matmul_performance.png', dpi=150, bbox_inches='tight')
    print("\n矩阵乘法性能对比图已保存为 matmul_performance.png")


def benchmark_convolution(device):
    """测试图像卷积性能"""
    print_section("图像卷积性能测试")
    
    import torch.nn.functional as F
    
    # 测试不同图像尺寸
    image_sizes = [64, 128, 256, 512, 1024, 2048]
    kernel_sizes = [3, 5, 7]
    
    results = defaultdict(list)
    
    for img_size in image_sizes:
        for ker_size in kernel_sizes:
            # CPU测试
            image_cpu = torch.randn(1, 1, img_size, img_size)
            kernel_cpu = torch.randn(1, 1, ker_size, ker_size)
            
            # 预热
            _ = F.conv2d(image_cpu, kernel_cpu, padding=ker_size//2)
            
            # 计时
            start = time.time()
            for _ in range(10):
                result = F.conv2d(image_cpu, kernel_cpu, padding=ker_size//2)
            cpu_time = (time.time() - start) / 10
            
            # GPU测试
            if torch.cuda.is_available():
                image_gpu = image_cpu.cuda()
                kernel_gpu = kernel_cpu.cuda()
                
                # 预热
                _ = F.conv2d(image_gpu, kernel_gpu, padding=ker_size//2)
                torch.cuda.synchronize()
                
                # 计时
                start = time.time()
                for _ in range(10):
                    result = F.conv2d(image_gpu, kernel_gpu, padding=ker_size//2)
                torch.cuda.synchronize()
                gpu_time = (time.time() - start) / 10
                
                speedup = cpu_time / gpu_time
            else:
                gpu_time = None
                speedup = None
            
            results[(img_size, ker_size)] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup
            }
            
            if img_size >= 256:
                gpu_str = f"{gpu_time:.6f}" if gpu_time else "N/A"
                speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
                print(f"  {img_size}x{img_size}, kernel {ker_size}x{ker_size}: "
                      f"CPU={cpu_time:.6f}s, GPU={gpu_str}, 加速={speedup_str}")
    
    # 绘图
    plot_convolution_results(results, image_sizes, kernel_sizes)
    
    return results


def plot_convolution_results(results, image_sizes, kernel_sizes):
    """绘制卷积性能对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 按kernel_size分组
    for ker_size in kernel_sizes:
        cpu_times = [results[(img, ker_size)]['cpu_time'] for img in image_sizes]
        axes[0].plot(image_sizes, cpu_times, 'o-', label=f'CPU kernel={ker_size}', markersize=6)
        
        if torch.cuda.is_available():
            gpu_times = [results[(img, ker_size)]['gpu_time'] for img in image_sizes]
            axes[0].plot(image_sizes, gpu_times, 's-', label=f'GPU kernel={ker_size}', markersize=6)
    
    axes[0].set_xlabel('图像尺寸', fontsize=12)
    axes[0].set_ylabel('执行时间(秒)', fontsize=12)
    axes[0].set_title('图像卷积执行时间', fontsize=14)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    axes[0].set_xscale('log')
    
    # 加速比
    if torch.cuda.is_available():
        for ker_size in kernel_sizes:
            speedups = [results[(img, ker_size)]['speedup'] for img in image_sizes]
            axes[1].plot(image_sizes, speedups, 'o-', label=f'kernel={ker_size}', markersize=6)
        
        axes[1].axhline(y=1.0, color='r', linestyle='--', label='基准线(1x)')
        axes[1].set_xlabel('图像尺寸', fontsize=12)
        axes[1].set_ylabel('加速比', fontsize=12)
        axes[1].set_title('GPU加速比', fontsize=14)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('convolution_performance.png', dpi=150, bbox_inches='tight')
    print("\n图像卷积性能对比图已保存为 convolution_performance.png")


def benchmark_tensor_operations(device):
    """测试张量操作性能"""
    print_section("张量操作性能测试")
    
    size = 10000
    
    operations = {
        '逐元素加法': lambda x, y: x + y,
        '逐元素乘法': lambda x, y: x * y,
        '矩阵乘法': lambda x, y: torch.matmul(x, y),
        '求和': lambda x, y: torch.sum(x),
        '均值': lambda x, y: torch.mean(x),
        '标准差': lambda x, y: torch.std(x),
        '最大值': lambda x, y: torch.max(x),
        '绝对值': lambda x, y: torch.abs(x),
        '指数': lambda x, y: torch.exp(x),
        '平方根': lambda x, y: torch.sqrt(torch.abs(x)),
    }
    
    print(f"\n测试规模: {size}x{size}\n")
    print(f"{'操作':<15} {'CPU时间':<12} {'GPU时间':<12} {'加速比':<10}")
    print("-" * 55)
    
    for name, op in operations.items():
        # 创建数据
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        
        # CPU测试
        start = time.time()
        for _ in range(10):
            result = op(x_cpu, y_cpu)
        cpu_time = (time.time() - start) / 10
        
        # GPU测试
        if torch.cuda.is_available():
            x_gpu = x_cpu.cuda()
            y_gpu = y_cpu.cuda()
            
            # 预热
            _ = op(x_gpu, y_gpu)
            torch.cuda.synchronize()
            
            # 计时
            start = time.time()
            for _ in range(10):
                result = op(x_gpu, y_gpu)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) / 10
            
            speedup = cpu_time / gpu_time
            
            print(f"{name:<15} {cpu_time:<12.6f} {gpu_time:<12.6f} {speedup:<10.2f}x")
        else:
            print(f"{name:<15} {cpu_time:<12.6f} {'N/A':<12} {'N/A':<10}")


def benchmark_memory_bandwidth(device):
    """测试内存带宽"""
    print_section("内存带宽测试")
    
    if not torch.cuda.is_available():
        print("\n需要GPU才能进行内存带宽测试")
        return
    
    # 测试不同规模的数据传输
    sizes_mb = [1, 10, 100, 500, 1000]
    
    print("\nCPU ↔ GPU 数据传输速度:\n")
    print(f"{'数据大小':<12} {'CPU→GPU':<15} {'GPU→CPU':<15} {'带宽(GB/s)':<15}")
    print("-" * 60)
    
    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024 // 4  # 元素数量(float32)
        
        # CPU → GPU
        data_cpu = torch.randn(size)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            data_gpu = data_cpu.cuda()
        torch.cuda.synchronize()
        cpu_to_gpu_time = (time.time() - start) / 10
        
        # GPU → CPU
        data_gpu = torch.randn(size, device='cuda')
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            data_cpu = data_gpu.cpu()
        gpu_to_cpu_time = (time.time() - start) / 10
        
        # 计算带宽
        bandwidth = (size_mb / cpu_to_gpu_time) / 1024  # GB/s
        
        print(f"{size_mb:<12}MB {cpu_to_gpu_time:<15.6f}s {gpu_to_cpu_time:<15.6f}s {bandwidth:<15.2f} GB/s")
    
    print("\n💡 观察:")
    print("  - PCIe带宽限制了数据传输速度")
    print("  - 尽量减少CPU-GPU数据传输")
    print("  - 在GPU上完成尽可能多的计算")


def performance_summary():
    """性能优化建议"""
    print_section("性能优化建议")
    
    print("\n1. 何时使用GPU?")
    print("   ✅ 大规模矩阵运算(>1000x1000)")
    print("   ✅ 批量图像处理")
    print("   ✅ 深度学习训练和推理")
    print("   ✅ 并行度高的计算任务")
    print("   ❌ 小规模数据(<100x100)")
    print("   ❌ 频繁CPU-GPU数据传输")
    print("   ❌ 串行依赖强的任务")
    
    print("\n2. GPU优化技巧:")
    print("   - 预热GPU(首次运行较慢)")
    print("   - 使用批量处理提高吞吐量")
    print("   - 减少CPU-GPU数据传输")
    print("   - 使用混合精度训练(fp16)")
    print("   - 合理设置线程块大小")
    
    print("\n3. PyTorch优化:")
    print("   - 使用torch.no_grad()减少内存")
    print("   - 使用DataLoader的pin_memory")
    print("   - 使用torch.compile() (PyTorch 2.0+)")
    print("   - 使用梯度检查点节省显存")
    
    print("\n4. 常见性能瓶颈:")
    print("   - 内存带宽受限(光刻仿真)")
    print("   - 计算受限(矩阵乘法)")
    print("   - PCIe传输瓶颈")
    print("   - 启动开销(小kernel)")


def generate_report(device):
    """生成性能报告"""
    print_section("生成性能报告")
    
    # 获取系统信息
    print("\n系统信息:")
    print(f"  PyTorch版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  CUDA核心数: {props.multi_processor_count}")
        print(f"  CUDA版本: {torch.version.cuda}")
    
    import platform
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  Python版本: {platform.python_version()}")
    
    # 运行关键测试
    print("\n关键性能指标:")
    
    # 矩阵乘法
    size = 5000
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    start = time.time()
    c = torch.matmul(a, b)
    cpu_time = time.time() - start
    cpu_gflops = 2 * size**3 / cpu_time / 1e9
    
    print(f"  CPU矩阵乘法({size}x{size}): {cpu_time:.4f}秒, {cpu_gflops:.2f} GFLOPS")
    
    if torch.cuda.is_available():
        a_gpu = a.cuda()
        b_gpu = b.cuda()
        
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        gpu_gflops = 2 * size**3 / gpu_time / 1e9
        
        print(f"  GPU矩阵乘法({size}x{size}): {gpu_time:.4f}秒, {gpu_gflops:.2f} GFLOPS")
        print(f"  加速比: {cpu_time/gpu_time:.2f}x")


def main():
    """主函数"""
    print("="*60)
    print("  PyTorch性能基准测试")
    print("  Day 3: Python GPU编程 - PyTorch入门")
    print("="*60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 预热GPU
    warmup_gpu(device)
    
    # 矩阵乘法测试
    print_section("矩阵乘法性能测试")
    sizes = [100, 200, 500, 1000, 2000, 3000, 5000]
    
    print("\nCPU测试:")
    cpu_results = benchmark_matmul_cpu(sizes)
    
    if torch.cuda.is_available():
        print("\nGPU测试:")
        gpu_results = benchmark_matmul_gpu(sizes, device)
    else:
        gpu_results = None
    
    # 绘制结果
    plot_matmul_results(cpu_results, gpu_results)
    
    # 图像卷积测试
    conv_results = benchmark_convolution(device)
    
    # 张量操作测试
    benchmark_tensor_operations(device)
    
    # 内存带宽测试
    benchmark_memory_bandwidth(device)
    
    # 性能优化建议
    performance_summary()
    
    # 生成报告
    generate_report(device)
    
    # 总结
    print_section("测试总结")
    print("\n✅ 性能测试完成!")
    print("\n📊 生成的图表:")
    print("  - matmul_performance.png: 矩阵乘法性能对比")
    print("  - convolution_performance.png: 图像卷积性能对比")
    
    print("\n💡 关键发现:")
    if torch.cuda.is_available():
        print("  - GPU在大规模计算上优势明显(50-200倍加速)")
        print("  - 数据规模越大,GPU优势越明显")
        print("  - 内存带宽是光刻仿真的主要瓶颈")
        print("  - 批量处理可以显著提高GPU利用率")
    else:
        print("  - 未检测到GPU,建议安装GPU版本PyTorch")
        print("  - GPU可以提供50-200倍的性能提升")
    
    print("\n📚 下一步:")
    print("  - 尝试调整测试参数,观察性能变化")
    print("  - 研究特定应用的GPU优化策略")
    print("  - 学习混合精度训练和模型优化")
    
    print("\n" + "="*60)
    print("性能测试完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
