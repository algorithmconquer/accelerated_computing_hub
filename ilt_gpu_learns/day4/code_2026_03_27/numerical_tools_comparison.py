"""
数值计算工具对比程序

本程序对比不同数值计算工具的性能,包括:
1. NumPy vs PyTorch (CPU)
2. PyTorch CPU vs GPU
3. 不同规模的性能分析
4. 工具选择建议

作者: GPU ILT学习体系
日期: 2026-03-27
"""

import numpy as np
import torch
import time


def check_gpu():
    """检查GPU环境"""
    print("="*60)
    print("GPU环境检查")
    print("="*60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU型号: {gpu_name}")
        print(f"GPU显存: {total_memory:.2f} GB")
    else:
        device = torch.device('cpu')
        print("未检测到GPU,将使用CPU运行")
    
    print()
    return device


def matrix_multiplication_benchmark(device):
    """
    矩阵乘法性能对比
    
    这是最基础的线性代数运算,在光刻仿真中大量使用
    """
    print("="*60)
    print("1. 矩阵乘法性能对比")
    print("="*60)
    
    sizes = [100, 200, 500, 1000, 2000, 4000]
    
    print(f"\n测试矩阵大小: {sizes}")
    print(f"\n{'Size':<10} {'NumPy(ms)':<12} {'PyTorch-CPU(ms)':<16} {'PyTorch-GPU(ms)':<16} {'GPU加速比':<10}")
    print("-" * 70)
    
    results = []
    
    for size in sizes:
        # NumPy (CPU)
        a_np = np.random.randn(size, size).astype(np.float32)
        b_np = np.random.randn(size, size).astype(np.float32)
        
        start = time.time()
        c_np = np.matmul(a_np, b_np)
        time_numpy = (time.time() - start) * 1000
        
        # PyTorch CPU
        a_torch_cpu = torch.from_numpy(a_np)
        b_torch_cpu = torch.from_numpy(b_np)
        
        start = time.time()
        c_torch_cpu = torch.matmul(a_torch_cpu, b_torch_cpu)
        time_torch_cpu = (time.time() - start) * 1000
        
        # PyTorch GPU
        if device.type == 'cuda':
            a_torch_gpu = a_torch_cpu.cuda()
            b_torch_gpu = b_torch_cpu.cuda()
            
            # 预热
            _ = torch.matmul(a_torch_gpu, b_torch_gpu)
            torch.cuda.synchronize()
            
            start = time.time()
            c_torch_gpu = torch.matmul(a_torch_gpu, b_torch_gpu)
            torch.cuda.synchronize()
            time_torch_gpu = (time.time() - start) * 1000
            
            speedup = time_numpy / time_torch_gpu
        else:
            time_torch_gpu = float('nan')
            speedup = float('nan')
        
        print(f"{size:<10} {time_numpy:<12.2f} {time_torch_cpu:<16.2f} {time_torch_gpu:<16.2f} {speedup:<10.2f}x")
        
        results.append({
            'size': size,
            'numpy': time_numpy,
            'torch_cpu': time_torch_cpu,
            'torch_gpu': time_torch_gpu,
            'speedup': speedup
        })
    
    return results


def fft_benchmark(device):
    """
    FFT性能对比
    
    FFT是光刻仿真的核心操作
    """
    print("\n" + "="*60)
    print("2. FFT性能对比")
    print("="*60)
    
    sizes = [64, 128, 256, 512, 1024, 2048]
    
    print(f"\n测试矩阵大小: {sizes}")
    print(f"\n{'Size':<10} {'NumPy(ms)':<12} {'PyTorch-CPU(ms)':<16} {'PyTorch-GPU(ms)':<16} {'GPU加速比':<10}")
    print("-" * 70)
    
    results = []
    
    for size in sizes:
        # NumPy (CPU)
        a_np = np.random.randn(size, size).astype(np.float32)
        
        start = time.time()
        fft_np = np.fft.fft2(a_np)
        time_numpy = (time.time() - start) * 1000
        
        # PyTorch CPU
        a_torch_cpu = torch.from_numpy(a_np)
        
        start = time.time()
        fft_torch_cpu = torch.fft.fft2(a_torch_cpu)
        time_torch_cpu = (time.time() - start) * 1000
        
        # PyTorch GPU
        if device.type == 'cuda':
            a_torch_gpu = a_torch_cpu.cuda()
            
            # 预热
            _ = torch.fft.fft2(a_torch_gpu)
            torch.cuda.synchronize()
            
            start = time.time()
            fft_torch_gpu = torch.fft.fft2(a_torch_gpu)
            torch.cuda.synchronize()
            time_torch_gpu = (time.time() - start) * 1000
            
            speedup = time_numpy / time_torch_gpu
        else:
            time_torch_gpu = float('nan')
            speedup = float('nan')
        
        print(f"{size:<10} {time_numpy:<12.2f} {time_torch_cpu:<16.2f} {time_torch_gpu:<16.2f} {speedup:<10.2f}x")
        
        results.append({
            'size': size,
            'numpy': time_numpy,
            'torch_cpu': time_torch_cpu,
            'torch_gpu': time_torch_gpu,
            'speedup': speedup
        })
    
    return results


def element_wise_operations_benchmark(device):
    """
    逐元素运算性能对比
    """
    print("\n" + "="*60)
    print("3. 逐元素运算性能对比")
    print("="*60)
    
    sizes = [10000, 100000, 1000000, 10000000]
    
    print(f"\n测试元素数量: {sizes}")
    print(f"\n{'Size':<12} {'NumPy(ms)':<12} {'PyTorch-CPU(ms)':<16} {'PyTorch-GPU(ms)':<16} {'GPU加速比':<10}")
    print("-" * 70)
    
    results = []
    
    for size in sizes:
        # NumPy (CPU)
        a_np = np.random.randn(size).astype(np.float32)
        b_np = np.random.randn(size).astype(np.float32)
        
        start = time.time()
        c_np = a_np * b_np + np.sin(a_np)  # 复杂运算
        time_numpy = (time.time() - start) * 1000
        
        # PyTorch CPU
        a_torch_cpu = torch.from_numpy(a_np)
        b_torch_cpu = torch.from_numpy(b_np)
        
        start = time.time()
        c_torch_cpu = a_torch_cpu * b_torch_cpu + torch.sin(a_torch_cpu)
        time_torch_cpu = (time.time() - start) * 1000
        
        # PyTorch GPU
        if device.type == 'cuda':
            a_torch_gpu = a_torch_cpu.cuda()
            b_torch_gpu = b_torch_cpu.cuda()
            
            # 预热
            _ = a_torch_gpu * b_torch_gpu + torch.sin(a_torch_gpu)
            torch.cuda.synchronize()
            
            start = time.time()
            c_torch_gpu = a_torch_gpu * b_torch_gpu + torch.sin(a_torch_gpu)
            torch.cuda.synchronize()
            time_torch_gpu = (time.time() - start) * 1000
            
            speedup = time_numpy / time_torch_gpu
        else:
            time_torch_gpu = float('nan')
            speedup = float('nan')
        
        print(f"{size:<12} {time_numpy:<12.2f} {time_torch_cpu:<16.2f} {time_torch_gpu:<16.2f} {speedup:<10.2f}x")
        
        results.append({
            'size': size,
            'numpy': time_numpy,
            'torch_cpu': time_torch_cpu,
            'torch_gpu': time_torch_gpu,
            'speedup': speedup
        })
    
    return results


def automatic_differentiation_test(device):
    """
    自动微分性能测试
    
    这是PyTorch的独特优势,NumPy不支持
    """
    print("\n" + "="*60)
    print("4. 自动微分性能测试")
    print("="*60)
    
    print("\n注意: NumPy不支持自动微分,只能手动实现")
    print("PyTorch提供强大的自动微分功能,这是ILT优化的核心")
    
    size = 1000
    
    # 创建需要梯度的张量
    x = torch.randn(size, size, device=device, requires_grad=True)
    
    # 前向传播
    start = time.time()
    y = x ** 2
    z = y.sum()
    forward_time = (time.time() - start) * 1000
    
    # 反向传播
    start = time.time()
    z.backward()
    backward_time = (time.time() - start) * 1000
    
    print(f"\n矩阵大小: {size}x{size}")
    print(f"前向传播时间: {forward_time:.2f} ms")
    print(f"反向传播时间: {backward_time:.2f} ms")
    print(f"总时间: {forward_time + backward_time:.2f} ms")
    
    # 验证梯度
    expected_grad = 2 * x
    error = torch.abs(x.grad - expected_grad).max()
    print(f"\n梯度验证: 最大误差 = {error:.8f} (应接近0)")
    
    return forward_time, backward_time


def memory_usage_test(device):
    """
    内存使用对比
    """
    print("\n" + "="*60)
    print("5. 内存使用分析")
    print("="*60)
    
    import sys
    
    # NumPy数组
    size = 10000
    a_np = np.random.randn(size, size).astype(np.float32)
    numpy_memory = a_np.nbytes / 1024**2  # MB
    
    # PyTorch CPU张量
    a_torch_cpu = torch.from_numpy(a_np)
    torch_cpu_memory = a_torch_cpu.element_size() * a_torch_cpu.nelement() / 1024**2
    
    # PyTorch GPU张量
    if device.type == 'cuda':
        a_torch_gpu = a_torch_cpu.cuda()
        
        # GPU显存使用
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        print(f"\n矩阵大小: {size}x{size}")
        print(f"\n内存使用:")
        print(f"  NumPy数组:     {numpy_memory:.2f} MB")
        print(f"  PyTorch CPU:   {torch_cpu_memory:.2f} MB")
        print(f"  PyTorch GPU:   {torch_cpu_memory:.2f} MB")
        print(f"\nGPU显存:")
        print(f"  已分配:        {allocated:.2f} MB")
        print(f"  已保留:        {reserved:.2f} MB")
        
        # GPU内存碎片
        print(f"\n内存效率: {allocated/reserved*100:.1f}%")
    else:
        print(f"\n矩阵大小: {size}x{size}")
        print(f"\n内存使用:")
        print(f"  NumPy数组:     {numpy_memory:.2f} MB")
        print(f"  PyTorch CPU:   {torch_cpu_memory:.2f} MB")


def tool_selection_guide():
    """
    工具选择指南
    """
    print("\n" + "="*60)
    print("6. 工具选择指南")
    print("="*60)
    
    print("""
╔══════════════════════════════════════════════════════════╗
║          数值计算工具选择决策树                          ║
╚══════════════════════════════════════════════════════════╝

开始
  │
  ├─ 需要GPU加速吗?
  │   ├─ 是 → PyTorch (GPU)
  │   └─ 否 → 继续
  │
  ├─ 需要自动微分吗?
  │   ├─ 是 → PyTorch (CPU或GPU)
  │   └─ 否 → 继续
  │
  ├─ 数据规模大吗(>1GB)?
  │   ├─ 是 → PyTorch (GPU推荐)
  │   └─ 否 → 继续
  │
  ├─ 已有NumPy代码吗?
  │   ├─ 是 → 保持NumPy
  │   └─ 否 → PyTorch (更现代)
  │
  └─ 简单原型/教学?
      ├─ 是 → NumPy (简单直观)
      └─ 否 → PyTorch (功能强大)


推荐使用场景:

【NumPy】
  ✓ 小规模数据处理
  ✓ 快速原型开发
  ✓ 教学和演示
  ✓ 已有NumPy代码库
  ✗ 不支持GPU
  ✗ 不支持自动微分

【PyTorch CPU】
  ✓ 需要自动微分
  ✓ 中等规模计算
  ✓ 开发和调试
  ✓ 无GPU环境
  ✗ 性能不如GPU

【PyTorch GPU】
  ✓ 大规模计算
  ✓ 性能要求高
  ✓ 深度学习应用
  ✓ GPU加速ILT仿真
  ✗ 需要NVIDIA GPU
  ✗ 需要CUDA环境


在GPU ILT项目中的建议:

1. 学习阶段(Day 1-10):
   - NumPy: 理解概念和原理
   - PyTorch CPU: 验证算法正确性

2. 开发阶段(Day 11-20):
   - PyTorch CPU: 调试和优化
   - PyTorch GPU: 性能测试

3. 生产阶段(Day 21-30):
   - PyTorch GPU: 大规模仿真
   - CUDA C++: 极致性能优化
""")


def performance_comparison_visualization(matmul_results, fft_results, element_results):
    """
    性能对比可视化
    """
    print("\n" + "="*60)
    print("7. 性能对比可视化")
    print("="*60)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 矩阵乘法
    sizes = [r['size'] for r in matmul_results]
    numpy_times = [r['numpy'] for r in matmul_results]
    torch_cpu_times = [r['torch_cpu'] for r in matmul_results]
    torch_gpu_times = [r['torch_gpu'] for r in matmul_results if not np.isnan(r['torch_gpu'])]
    
    axes[0].semilogy(sizes, numpy_times, 'o-', label='NumPy', linewidth=2, markersize=8)
    axes[0].semilogy(sizes, torch_cpu_times, 's-', label='PyTorch CPU', linewidth=2, markersize=8)
    if torch_gpu_times:
        axes[0].semilogy(sizes[:len(torch_gpu_times)], torch_gpu_times, '^-', label='PyTorch GPU', linewidth=2, markersize=8)
    axes[0].set_xlabel('Matrix Size', fontsize=12)
    axes[0].set_ylabel('Time (ms, log scale)', fontsize=12)
    axes[0].set_title('Matrix Multiplication Performance', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # FFT
    sizes = [r['size'] for r in fft_results]
    numpy_times = [r['numpy'] for r in fft_results]
    torch_cpu_times = [r['torch_cpu'] for r in fft_results]
    torch_gpu_times = [r['torch_gpu'] for r in fft_results if not np.isnan(r['torch_gpu'])]
    
    axes[1].semilogy(sizes, numpy_times, 'o-', label='NumPy', linewidth=2, markersize=8)
    axes[1].semilogy(sizes, torch_cpu_times, 's-', label='PyTorch CPU', linewidth=2, markersize=8)
    if torch_gpu_times:
        axes[1].semilogy(sizes[:len(torch_gpu_times)], torch_gpu_times, '^-', label='PyTorch GPU', linewidth=2, markersize=8)
    axes[1].set_xlabel('Matrix Size', fontsize=12)
    axes[1].set_ylabel('Time (ms, log scale)', fontsize=12)
    axes[1].set_title('FFT Performance', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 加速比
    matmul_speedups = [r['speedup'] for r in matmul_results if not np.isnan(r['speedup'])]
    fft_speedups = [r['speedup'] for r in fft_results if not np.isnan(r['speedup'])]
    
    x_pos = np.arange(len(matmul_speedups))
    width = 0.35
    
    axes[2].bar(x_pos - width/2, matmul_speedups, width, label='Matrix Multiplication', alpha=0.8)
    axes[2].bar(x_pos + width/2, fft_speedups[:len(matmul_speedups)], width, label='FFT', alpha=0.8)
    
    axes[2].set_xlabel('Size Index', fontsize=12)
    axes[2].set_ylabel('Speedup (x)', fontsize=12)
    axes[2].set_title('GPU Speedup over NumPy', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n性能对比图已保存为 performance_comparison.png")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Day 4: 数值计算工具对比程序")
    print("="*60)
    
    # 检查GPU
    device = check_gpu()
    
    # 1. 矩阵乘法性能对比
    matmul_results = matrix_multiplication_benchmark(device)
    
    # 2. FFT性能对比
    fft_results = fft_benchmark(device)
    
    # 3. 逐元素运算性能对比
    element_results = element_wise_operations_benchmark(device)
    
    # 4. 自动微分测试
    forward_time, backward_time = automatic_differentiation_test(device)
    
    # 5. 内存使用分析
    memory_usage_test(device)
    
    # 6. 工具选择指南
    tool_selection_guide()
    
    # 7. 可视化
    performance_comparison_visualization(matmul_results, fft_results, element_results)
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)
    print("\n关键结论:")
    print("  1. GPU在大规模计算上有显著优势(10-100倍加速)")
    print("  2. PyTorch提供自动微分,是ILT优化的利器")
    print("  3. NumPy适合小规模和原型开发")
    print("  4. 选择合适的工具能大幅提升开发效率")


if __name__ == "__main__":
    main()
