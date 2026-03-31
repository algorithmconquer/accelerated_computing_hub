#!/usr/bin/env python3
"""
stream_benchmark.py
CUDA流性能基准测试（Python版本）

功能：
1. 使用PyTorch实现流并发
2. 性能基准测试
3. 自动寻找最佳流数量
4. 性能可视化

运行: python stream_benchmark.py
"""

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def benchmark_single_stream(
    data: torch.Tensor, 
    iterations: int = 100
) -> float:
    """
    单流执行基准测试
    
    参数:
        data: 输入数据张量
        iterations: 计算迭代次数
    
    返回:
        平均执行时间（毫秒）
    """
    # 创建事件用于计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # 预热
    for _ in range(5):
        result = data * 1.01 + 0.1
    torch.cuda.synchronize()
    
    # 计时
    start.record()
    for _ in range(iterations):
        result = data * 1.01 + 0.1
        result = torch.sin(result) + torch.cos(result)
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def benchmark_multi_stream(
    data: torch.Tensor,
    num_streams: int,
    iterations: int = 100
) -> float:
    """
    多流执行基准测试
    
    参数:
        data: 输入数据张量
        num_streams: 流数量
        iterations: 计算迭代次数
    
    返回:
        平均执行时间（毫秒）
    """
    # 创建多个流
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    
    # 将数据分块
    chunk_size = data.shape[0] // num_streams
    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_streams)]
    
    # 预热
    with torch.cuda.stream(streams[0]):
        _ = chunks[0] * 1.01 + 0.1
    torch.cuda.synchronize()
    
    # 创建事件用于计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # 计时
    start.record()
    
    results = []
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            chunk = chunks[i]
            result = chunk.clone()
            for _ in range(iterations):
                result = result * 1.01 + 0.1
                result = torch.sin(result) + torch.cos(result)
            results.append(result)
    
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end)


def run_benchmark(
    data_size_mb: int = 64,
    stream_counts: List[int] = [1, 2, 4, 8, 16],
    num_runs: int = 10
) -> Tuple[List[float], List[float]]:
    """
    运行完整基准测试
    
    参数:
        data_size_mb: 数据大小（MB）
        stream_counts: 要测试的流数量列表
        num_runs: 每个配置运行次数
    
    返回:
        (平均时间列表, 加速比列表)
    """
    print("=== PyTorch CUDA流基准测试 ===\n")
    
    # 显示GPU信息
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}\n")
    
    # 测试配置
    print(f"测试配置:")
    print(f"  数据大小: {data_size_mb} MB")
    print(f"  流数量: {stream_counts}")
    print(f"  每配置运行次数: {num_runs}\n")
    
    # 创建数据
    num_elements = (data_size_mb * 1024 * 1024) // 4  # float32
    data = torch.randn(num_elements, device='cuda')
    
    results = {}
    
    # 测试每个流配置
    for num_streams in stream_counts:
        print(f"测试 {num_streams} 个流...", end=' ')
        times = []
        
        for _ in range(num_runs):
            if num_streams == 1:
                time_ms = benchmark_single_stream(data)
            else:
                time_ms = benchmark_multi_stream(data, num_streams)
            times.append(time_ms)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results[num_streams] = (avg_time, std_time)
        print(f"平均时间: {avg_time:.2f} ms (±{std_time:.2f})")
    
    # 计算加速比
    baseline_time = results[1][0]
    speedups = [baseline_time / results[n][0] for n in stream_counts]
    
    return [results[n][0] for n in stream_counts], speedups


def plot_results(
    stream_counts: List[int],
    times: List[float],
    speedups: List[float],
    output_file: str = 'stream_benchmark_results.png'
):
    """
    绘制性能对比图
    
    参数:
        stream_counts: 流数量列表
        times: 时间列表
        speedups: 加速比列表
        output_file: 输出文件名
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 执行时间
    ax1.bar(range(len(stream_counts)), times, color='steelblue', alpha=0.8)
    ax1.set_xlabel('流数量', fontsize=12)
    ax1.set_ylabel('执行时间 (ms)', fontsize=12)
    ax1.set_title('不同流数量下的执行时间', fontsize=14)
    ax1.set_xticks(range(len(stream_counts)))
    ax1.set_xticklabels(stream_counts)
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上标注数值
    for i, (count, time) in enumerate(zip(stream_counts, times)):
        ax1.text(i, time + 1, f'{time:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 图2: 加速比
    colors = ['green' if s >= 1.5 else 'orange' if s >= 1.2 else 'red' 
              for s in speedups]
    ax2.bar(range(len(stream_counts)), speedups, color=colors, alpha=0.8)
    ax2.axhline(y=1.0, color='red', linestyle='--', label='基准线', alpha=0.5)
    ax2.set_xlabel('流数量', fontsize=12)
    ax2.set_ylabel('加速比', fontsize=12)
    ax2.set_title('性能加速比（相对于单流）', fontsize=14)
    ax2.set_xticks(range(len(stream_counts)))
    ax2.set_xticklabels(stream_counts)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # 在柱子上标注数值
    for i, (count, speedup) in enumerate(zip(stream_counts, speedups)):
        ax2.text(i, speedup + 0.05, f'{speedup:.2f}x', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")


def find_optimal_streams(stream_counts: List[int], speedups: List[float]) -> Tuple[int, float]:
    """
    找到最佳流数量
    
    参数:
        stream_counts: 流数量列表
        speedups: 加速比列表
    
    返回:
        (最佳流数量, 最佳加速比)
    """
    best_idx = np.argmax(speedups)
    return stream_counts[best_idx], speedups[best_idx]


def main():
    """主函数"""
    # 运行基准测试
    stream_counts = [1, 2, 4, 8, 16]
    times, speedups = run_benchmark(
        data_size_mb=64,
        stream_counts=stream_counts,
        num_runs=10
    )
    
    # 绘制结果
    plot_results(stream_counts, times, speedups)
    
    # 找到最佳配置
    best_streams, best_speedup = find_optimal_streams(stream_counts, speedups)
    
    # 打印总结
    print("\n" + "="*50)
    print("性能总结:")
    print("="*50)
    for count, time, speedup in zip(stream_counts, times, speedups):
        print(f"{count:2d}个流: {time:6.2f} ms  (加速 {speedup:.2f}x)")
    print("="*50)
    print(f"\n最佳流数量: {best_streams}")
    print(f"最佳加速比: {best_speedup:.2f}x")
    
    # 给出建议
    print("\n建议:")
    if best_streams <= 4:
        print("  • 当前硬件最适合使用 2-4 个流")
    else:
        print("  • 当前硬件可以支持更多并发流")
    print("  • 建议配置: 4-8 个流以获得最佳性能平衡")
    print("  • 过多的流可能导致资源竞争")
    
    # 保存结果到文件
    with open('benchmark_summary.txt', 'w') as f:
        f.write("=== CUDA流性能基准测试结果 ===\n\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"CUDA版本: {torch.version.cuda}\n\n")
        f.write("配置性能:\n")
        for count, time, speedup in zip(stream_counts, times, speedups):
            f.write(f"  {count}个流: {time:.2f} ms ({speedup:.2f}x)\n")
        f.write(f"\n最佳配置: {best_streams}个流 ({best_speedup:.2f}x)\n")
    
    print("\n结果已保存到 benchmark_summary.txt")


if __name__ == "__main__":
    main()
