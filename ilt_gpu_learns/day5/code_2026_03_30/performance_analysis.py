"""
GPU性能分析工具演示

本脚本演示如何使用不同的工具分析GPU程序性能
包含：
1. CUDA Event计时
2. nvprof命令行分析
3. 性能指标计算
"""

import subprocess
import torch
import numpy as np
import time

def cuda_event_timer():
    """
    演示使用CUDA Event进行精确计时
    CUDA Event比CPU计时更准确，因为它直接测量GPU执行时间
    """
    print("=== 1. CUDA Event 计时示例 ===\n")
    
    # 创建CUDA事件
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # 创建测试数据
    size = 10_000_000
    a = torch.randn(size, device='cuda')
    b = torch.randn(size, device='cuda')
    c = torch.empty(size, device='cuda')
    
    # 记录开始时间
    start.record()
    
    # 执行GPU操作
    torch.add(a, b, out=c)
    
    # 记录结束时间
    end.record()
    
    # 等待GPU完成
    torch.cuda.synchronize()
    
    # 计算时间
    elapsed_time = start.elapsed_time(end)  # 毫秒
    
    print(f"数据大小: {size:,} 元素 ({size * 4 / 1024 / 1024:.2f} MB)")
    print(f"GPU执行时间: {elapsed_time:.3f} ms")
    
    # 计算内存带宽
    # 读取a和b，写入c，总共3*size个float
    bytes_transferred = 3 * size * 4  # 3个数组，每个4字节
    bandwidth = (bytes_transferred / elapsed_time) / 1e6  # GB/s
    print(f"内存带宽: {bandwidth:.2f} GB/s")
    print()


def nvprof_analysis():
    """
    演示如何使用nvprof进行性能分析
    nvprof是CUDA的性能分析工具，可以测量各种性能指标
    """
    print("=== 2. nvprof 性能分析 ===\n")
    
    print("nvprof常用命令:\n")
    
    commands = [
        # 基础性能分析
        ("基础分析", "nvprof ./program"),
        
        # 打印GPU跟踪信息
        ("详细跟踪", "nvprof --print-gpu-trace ./program"),
        
        # 分析内存访问效率
        ("内存效率", 
         "nvprof --metrics gld_efficiency,gst_efficiency ./program"),
        
        # 分析计算吞吐量
        ("计算吞吐量", 
         "nvprof --metrics flops_sp,flops_dp ./program"),
        
        # 分析占用率
        ("占用率", 
         "nvprof --metrics achieved_occupancy ./program"),
        
        # 分析内存带宽
        ("内存带宽", 
         "nvprof --metrics dram_read_throughput,dram_write_throughput ./program"),
        
        # 分析共享内存使用
        ("共享内存", 
         "nvprof --metrics shared_load_throughput,shared_store_throughput ./program"),
    ]
    
    for name, command in commands:
        print(f"{name:12s}: {command}")
    
    print("\n查看所有可用指标:")
    print("  nvprof --query-metrics")
    print()


def performance_metrics():
    """
    演示计算各种性能指标
    """
    print("=== 3. 性能指标计算 ===\n")
    
    # 示例：矩阵乘法性能分析
    print("示例：1024x1024 矩阵乘法\n")
    
    width = 1024
    time_ms = 18.67  # 实测时间（ms）
    
    # 1. 计算FLOPS（每秒浮点运算次数）
    # 矩阵乘法浮点运算次数 = 2 * width^3
    flops = 2 * width * width * width
    gflops = (flops / time_ms) / 1e6  # 转换为GFLOPS
    print(f"1. 计算性能 (FLOPS):")
    print(f"   浮点运算次数: {flops:,}")
    print(f"   性能: {gflops:.2f} GFLOPS")
    print()
    
    # 2. 计算内存带宽
    # 读取A矩阵：width * width
    # 读取B矩阵：width * width（重用多次，但理论最小带宽）
    # 写入C矩阵：width * width
    # 实际带宽计算需要考虑数据重用
    bytes_read_A = width * width * 4
    bytes_read_B = width * width * 4
    bytes_write_C = width * width * 4
    total_bytes = bytes_read_A + bytes_read_B + bytes_write_C
    
    bandwidth = (total_bytes / time_ms) / 1e6  # GB/s
    print(f"2. 内存带宽:")
    print(f"   数据传输量: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"   带宽: {bandwidth:.2f} GB/s")
    print()
    
    # 3. 计算占用率
    print("3. 占用率计算:")
    print("   公式: Occupancy = Active Warps / Max Warps per SM")
    print("   影响因素:")
    print("     - 线程块大小")
    print("     - 每个线程的寄存器数量")
    print("     - 每个线程块的共享内存数量")
    print("   使用工具: CUDA Occupancy Calculator")
    print()


def pytorch_profiler_demo():
    """
    演示PyTorch Profiler的使用
    PyTorch Profiler提供了更详细的分析功能
    """
    print("=== 4. PyTorch Profiler 示例 ===\n")
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # 执行一些GPU操作
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        c = torch.matmul(a, b)
        d = torch.sum(c)
    
    # 打印性能报告
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print()


def benchmark_best_practices():
    """
    演示GPU性能测试的最佳实践
    """
    print("=== 5. 性能测试最佳实践 ===\n")
    
    print("1. 预热GPU")
    print("   - 第一次运行会包含初始化开销")
    print("   - 运行几次预热，然后测量稳定性能\n")
    
    # 示例
    a = torch.randn(1000000, device='cuda')
    b = torch.randn(1000000, device='cuda')
    
    # 预热
    for _ in range(10):
        c = a + b
    torch.cuda.synchronize()
    
    print("2. 多次测量取平均值")
    print("   - 单次测量可能受系统调度影响")
    print("   - 运行多次取平均值\n")
    
    # 测量
    times = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        c = a + b
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"   平均时间: {avg_time:.4f} ms")
    print(f"   标准差: {std_time:.4f} ms")
    print()
    
    print("3. 使用CUDA Event计时")
    print("   - 比CPU计时更准确")
    print("   - 直接测量GPU执行时间")
    print()
    
    print("4. 检查结果正确性")
    print("   - 性能优化的前提是结果正确")
    print("   - 每次优化后都要验证结果")
    print()


def analyze_kernel_performance():
    """
    演示如何分析CUDA核函数性能
    """
    print("=== 6. 核函数性能分析框架 ===\n")
    
    print("分析步骤:\n")
    
    print("1. 识别瓶颈")
    print("   命令: nvprof --print-gpu-trace ./program")
    print("   找出耗时最长的核函数\n")
    
    print("2. 分析内存访问")
    print("   命令: nvprof --metrics gld_efficiency,gst_efficiency ./program")
    print("   目标: 内存效率 > 80%\n")
    
    print("3. 分析占用率")
    print("   命令: nvprof --metrics achieved_occupancy ./program")
    print("   目标: 占用率 > 50%\n")
    
    print("4. 分析计算吞吐量")
    print("   命令: nvprof --metrics flops_sp ./program")
    print("   对比峰值性能\n")
    
    print("5. 优化建议:")
    optimizations = [
        "内存效率低 → 优化访问模式，使用共享内存",
        "占用率低 → 减少寄存器使用，调整线程块大小",
        "计算吞吐量低 → 循环展开，向量化",
        "带宽利用率低 → 合并访问，向量化加载",
    ]
    for opt in optimizations:
        print(f"   - {opt}")
    print()


def main():
    print("=" * 60)
    print("GPU性能分析工具演示")
    print("=" * 60)
    print()
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print()
    
    # 运行各项演示
    cuda_event_timer()
    nvprof_analysis()
    performance_metrics()
    pytorch_profiler_demo()
    benchmark_best_practices()
    analyze_kernel_performance()
    
    print("=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 运行 CUDA 程序: ./matrix_mul")
    print("2. 使用 nvprof 分析: nvprof ./matrix_mul")
    print("3. 查看详细指标: nvprof --print-gpu-trace ./matrix_mul")
    print("4. 对比优化前后的性能差异")


if __name__ == "__main__":
    main()
