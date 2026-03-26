#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: PyTorch自动微分示例
=========================

本脚本演示PyTorch的自动微分机制,包括:
1. 梯度计算基础
2. 计算图原理
3. 优化问题求解
4. 多变量优化
5. 神经网络反向传播

自动微分是深度学习和优化算法的核心技术。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def print_section(title):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def basic_gradient():
    """演示基础梯度计算"""
    print_section("1. 梯度计算基础")
    
    print("\n示例1: 简单的一元函数")
    print("函数: y = x^2")
    print("导数: dy/dx = 2x")
    
    # 创建需要梯度的张量
    x = torch.tensor([2.0], requires_grad=True)
    print(f"\n初始值: x = {x.item()}")
    
    # 前向传播:计算y
    y = x ** 2
    print(f"前向传播: y = x^2 = {y.item()}")
    
    # 反向传播:计算梯度
    y.backward()
    print(f"反向传播: dy/dx = {x.grad.item()}")
    print(f"验证: 2x = {2 * x.item()}")
    
    print("\n" + "-"*60)
    
    print("\n示例2: 复合函数")
    print("函数: z = (x^2 + 3x + 1)^2")
    print("导数: dz/dx = 2(x^2 + 3x + 1) * (2x + 3)")
    
    x = torch.tensor([2.0], requires_grad=True)
    print(f"\n初始值: x = {x.item()}")
    
    # 前向传播
    y = x**2 + 3*x + 1
    z = y ** 2
    
    print(f"前向传播: y = {y.item():.4f}, z = {z.item():.4f}")
    
    # 反向传播
    z.backward()
    
    print(f"反向传播: dz/dx = {x.grad.item():.4f}")
    
    # 手动验证
    manual_grad = 2 * (x.item()**2 + 3*x.item() + 1) * (2*x.item() + 3)
    print(f"手动验证: dz/dx = {manual_grad:.4f}")
    
    print("\n💡 关键概念:")
    print("  - requires_grad=True: 标记需要计算梯度的张量")
    print("  - backward(): 反向传播,计算梯度")
    print("  - .grad: 存储计算得到的梯度")


def computation_graph():
    """演示计算图"""
    print_section("2. 计算图原理")
    
    print("\n计算图是自动微分的数据结构基础")
    print("它记录了张量之间的运算关系\n")
    
    # 创建计算图
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)
    
    print("构建计算图:")
    print(f"  x = {x.item()}, y = {y.item()}")
    
    # 前向传播
    a = x + y          # a = x + y
    b = x * y          # b = x * y
    c = a * b          # c = (x+y) * (x*y)
    z = c ** 2         # z = [(x+y) * (x*y)]^2
    
    print(f"\n前向传播:")
    print(f"  a = x + y = {a.item():.4f}")
    print(f"  b = x * y = {b.item():.4f}")
    print(f"  c = a * b = {c.item():.4f}")
    print(f"  z = c^2 = {z.item():.4f}")
    
    # 查看计算图
    print(f"\n计算图结构:")
    print(f"  z.grad_fn: {z.grad_fn}")
    print(f"  c.grad_fn: {c.grad_fn}")
    print(f"  a.grad_fn: {a.grad_fn}")
    print(f"  b.grad_fn: {b.grad_fn}")
    
    # 反向传播
    z.backward()
    
    print(f"\n反向传播结果:")
    print(f"  dz/dx = {x.grad.item():.4f}")
    print(f"  dz/dy = {y.grad.item():.4f}")
    
    # 手动验证
    # z = [(x+y) * xy]^2
    # dz/dx = 2[(x+y)xy] * [(x+y)y + xy]
    x_val, y_val = 1.0, 2.0
    manual_dx = 2 * (x_val + y_val) * x_val * y_val * ((x_val + y_val) * y_val + x_val * y_val)
    manual_dy = 2 * (x_val + y_val) * x_val * y_val * ((x_val + y_val) * x_val + x_val * y_val)
    
    print(f"\n手动验证:")
    print(f"  dz/dx (手动) = {manual_dx:.4f}")
    print(f"  dz/dy (手动) = {manual_dy:.4f}")
    
    print("\n💡 计算图的作用:")
    print("  1. 记录运算关系,实现自动微分")
    print("  2. 支持动态图(PyTorch默认),更灵活")
    print("  3. 可以查看grad_fn追溯计算历史")


def gradient_descent_simple():
    """演示简单的梯度下降优化"""
    print_section("3. 梯度下降优化")
    
    print("\n目标: 找到使 f(x) = (x-3)^2 + (x+1)^2 最小的x")
    print("理论解: f'(x) = 2(x-3) + 2(x+1) = 4x - 4 = 0 => x = 1\n")
    
    # 初始化
    x = torch.tensor([0.0], requires_grad=True)
    learning_rate = 0.1
    num_iterations = 20
    
    # 记录优化过程
    x_history = [x.item()]
    loss_history = []
    
    print(f"初始值: x = {x.item():.4f}")
    print(f"学习率: {learning_rate}")
    print(f"\n优化过程:")
    
    for i in range(num_iterations):
        # 前向传播:计算损失
        loss = (x - 3)**2 + (x + 1)**2
        loss_history.append(loss.item())
        
        # 反向传播:计算梯度
        loss.backward()
        
        # 更新参数(梯度下降)
        with torch.no_grad():
            x -= learning_rate * x.grad
        
        # 清零梯度
        x.grad.zero_()
        
        # 记录
        x_history.append(x.item())
        
        if i % 5 == 0:
            print(f"  迭代{i:2d}: x = {x.item():.4f}, loss = {loss.item():.4f}")
    
    print(f"\n最终结果: x = {x.item():.4f}")
    print(f"理论最优解: x = 1.0000")
    print(f"误差: {abs(x.item() - 1.0):.6f}")
    
    # 可视化
    visualize_optimization_1d(x_history, loss_history)
    
    print("\n💡 关键步骤:")
    print("  1. loss.backward() - 计算梯度")
    print("  2. with torch.no_grad() - 更新时不记录梯度")
    print("  3. x.grad.zero_() - 清零梯度,防止累加")


def visualize_optimization_1d(x_history, loss_history):
    """可视化一维优化过程"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图:目标函数和优化路径
    x_range = torch.linspace(-1, 3, 100)
    y_range = (x_range - 3)**2 + (x_range + 1)**2
    
    ax1.plot(x_range.numpy(), y_range.numpy(), 'b-', linewidth=2, label='目标函数')
    ax1.plot(x_history, [(x-3)**2 + (x+1)**2 for x in x_history], 
             'ro-', markersize=8, label='优化路径')
    ax1.axvline(x=1.0, color='g', linestyle='--', label='最优解 x=1')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('梯度下降优化路径', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图:损失收敛曲线
    ax2.plot(loss_history, 'b-', linewidth=2, marker='o', markersize=6)
    ax2.axhline(y=4.0, color='r', linestyle='--', label='最小损失=4')
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('损失值', fontsize=12)
    ax2.set_title('损失收敛曲线', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_1d.png', dpi=150, bbox_inches='tight')
    print("\n优化过程图已保存为 optimization_1d.png")


def multi_variable_optimization():
    """演示多变量优化"""
    print_section("4. 多变量优化")
    
    print("\n目标: 找到使 f(x,y) = (x-3)^2 + 2(y+1)^2 最小的(x,y)")
    print("理论解: x=3, y=-1, 最小值=0\n")
    
    # 初始化
    x = torch.tensor([0.0], requires_grad=True)
    y = torch.tensor([0.0], requires_grad=True)
    learning_rate = 0.1
    num_iterations = 50
    
    # 记录优化过程
    x_history = [x.item()]
    y_history = [y.item()]
    loss_history = []
    
    print(f"初始值: x = {x.item():.4f}, y = {y.item():.4f}")
    print(f"学习率: {learning_rate}")
    print(f"\n优化过程:")
    
    for i in range(num_iterations):
        # 前向传播:计算损失
        loss = (x - 3)**2 + 2*(y + 1)**2
        loss_history.append(loss.item())
        
        # 反向传播:计算梯度
        loss.backward()
        
        # 更新参数
        with torch.no_grad():
            x -= learning_rate * x.grad
            y -= learning_rate * y.grad
        
        # 清零梯度
        x.grad.zero_()
        y.grad.zero_()
        
        # 记录
        x_history.append(x.item())
        y_history.append(y.item())
        
        if i % 10 == 0:
            print(f"  迭代{i:2d}: x = {x.item():.4f}, y = {y.item():.4f}, loss = {loss.item():.4f}")
    
    print(f"\n最终结果: x = {x.item():.4f}, y = {y.item():.4f}")
    print(f"理论最优解: x = 3.0000, y = -1.0000")
    print(f"误差: Δx = {abs(x.item() - 3.0):.6f}, Δy = {abs(y.item() + 1.0):.6f}")
    
    # 可视化
    visualize_optimization_2d(x_history, y_history, loss_history)


def visualize_optimization_2d(x_history, y_history, loss_history):
    """可视化二维优化过程"""
    fig = plt.figure(figsize=(15, 5))
    
    # 左图:等高线图
    ax1 = fig.add_subplot(131)
    x_range = torch.linspace(-1, 5, 100)
    y_range = torch.linspace(-3, 1, 100)
    X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
    Z = (X - 3)**2 + 2*(Y + 1)**2
    
    contour = ax1.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=20, cmap='viridis')
    ax1.plot(x_history, y_history, 'ro-', markersize=6, linewidth=2, label='优化路径')
    ax1.plot(3, -1, 'g*', markersize=15, label='最优解')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('等高线图与优化路径', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax1, label='f(x,y)')
    
    # 中图:3D表面图
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis', alpha=0.6)
    ax2.plot(x_history, y_history, loss_history, 'ro-', markersize=4, linewidth=2, label='优化路径')
    ax2.set_xlabel('x', fontsize=10)
    ax2.set_ylabel('y', fontsize=10)
    ax2.set_zlabel('f(x,y)', fontsize=10)
    ax2.set_title('3D目标函数', fontsize=14)
    
    # 右图:损失收敛曲线
    ax3 = fig.add_subplot(133)
    ax3.plot(loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=0, color='r', linestyle='--', label='最小损失=0')
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('损失值', fontsize=12)
    ax3.set_title('损失收敛曲线', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('optimization_2d.png', dpi=150, bbox_inches='tight')
    print("\n优化过程图已保存为 optimization_2d.png")


def gradient_accumulation():
    """演示梯度累加问题"""
    print_section("5. 梯度累加问题")
    
    print("\n⚠️ 重要:PyTorch默认会累加梯度!\n")
    
    # 示例:梯度累加
    x = torch.tensor([1.0], requires_grad=True)
    
    print("示例:不清理梯度")
    for i in range(3):
        y = x ** 2
        y.backward()
        print(f"  第{i+1}次: x.grad = {x.grad.item():.4f}")
        # 没有清零梯度!
    
    print("\n示例:正确清理梯度")
    x = torch.tensor([1.0], requires_grad=True)
    
    for i in range(3):
        y = x ** 2
        y.backward()
        print(f"  第{i+1}次: x.grad = {x.grad.item():.4f}")
        x.grad.zero_()  # 清零梯度
    
    print("\n💡 解决方案:")
    print("  在每次迭代后调用 x.grad.zero_()")
    print("  或使用 optimizer.zero_grad()")


def learning_rate_effect():
    """演示学习率的影响"""
    print_section("6. 学习率的影响")
    
    print("\n对比不同学习率的优化效果\n")
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    num_iterations = 30
    
    plt.figure(figsize=(12, 8))
    
    for lr in learning_rates:
        x = torch.tensor([0.0], requires_grad=True)
        x_history = [x.item()]
        
        for i in range(num_iterations):
            loss = (x - 3)**2 + (x + 1)**2
            loss.backward()
            
            with torch.no_grad():
                x -= lr * x.grad
            
            x.grad.zero_()
            x_history.append(x.item())
        
        plt.plot(x_history, 'o-', label=f'lr={lr}', markersize=5)
        print(f"学习率 {lr}: 最终 x = {x.item():.4f}")
    
    plt.axhline(y=1.0, color='black', linestyle='--', label='最优解 x=1')
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('x值', fontsize=12)
    plt.title('不同学习率的优化效果对比', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_rate_comparison.png', dpi=150, bbox_inches='tight')
    print("\n学习率对比图已保存为 learning_rate_comparison.png")
    
    print("\n💡 观察:")
    print("  - 学习率太小:收敛慢,需要更多迭代")
    print("  - 学习率合适:快速收敛到最优解")
    print("  - 学习率太大:可能震荡甚至发散")


def gpu_optimization():
    """演示GPU上的优化"""
    print_section("7. GPU上的优化")
    
    if not torch.cuda.is_available():
        print("\n未检测到GPU,跳过此部分")
        return
    
    device = torch.device('cuda')
    print(f"\n使用设备: {device}")
    
    # 大规模优化问题
    n = 1000  # 变量数量
    
    print(f"\n优化问题: 最小化 ||Ax - b||^2")
    print(f"其中 A 是 {n}x{n} 矩阵, b 是 {n} 维向量\n")
    
    # 创建数据
    A = torch.randn(n, n, device=device)
    b = torch.randn(n, device=device)
    x = torch.randn(n, device=device, requires_grad=True)
    
    learning_rate = 0.001
    num_iterations = 100
    
    print("开始优化...")
    
    # 记录时间
    import time
    start_time = time.time()
    
    for i in range(num_iterations):
        # 前向传播
        residual = A @ x - b
        loss = torch.sum(residual ** 2)
        
        # 反向传播
        loss.backward()
        
        # 更新
        with torch.no_grad():
            x -= learning_rate * x.grad
        
        x.grad.zero_()
        
        if i % 20 == 0:
            print(f"  迭代{i:3d}: loss = {loss.item():.6f}")
    
    gpu_time = time.time() - start_time
    
    final_loss = torch.sum((A @ x - b) ** 2).item()
    print(f"\n最终损失: {final_loss:.6f}")
    print(f"GPU优化时间: {gpu_time:.4f}秒")
    
    print("\n💡 GPU优势:")
    print("  - 大规模矩阵运算显著加速")
    print("  - 自动微分在GPU上并行执行")
    print("  - 适合深度学习和大规模优化")


def practical_tips():
    """实用技巧"""
    print_section("8. 实用技巧")
    
    print("\n技巧1: 使用torch.no_grad()节省内存")
    print("  推理时不需要梯度,使用:")
    print("    with torch.no_grad():")
    print("        output = model(input)")
    
    print("\n技巧2: 梯度裁剪防止梯度爆炸")
    x = torch.randn(10, requires_grad=True)
    y = x.sum()
    y.backward()
    print(f"  原始梯度: {x.grad}")
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(x, max_norm=1.0)
    print(f"  裁剪后梯度: {x.grad}")
    
    print("\n技巧3: 查看梯度统计信息")
    x = torch.randn(100, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    
    print(f"  梯度均值: {x.grad.mean().item():.6f}")
    print(f"  梯度标准差: {x.grad.std().item():.6f}")
    print(f"  梯度最大值: {x.grad.max().item():.6f}")
    print(f"  梯度最小值: {x.grad.min().item():.6f}")
    
    print("\n技巧4: 检查梯度")
    def check_gradient(x, eps=1e-5):
        """数值梯度检查"""
        x_val = x.item()
        
        # 数值梯度
        x_plus = torch.tensor([x_val + eps])
        x_minus = torch.tensor([x_val - eps])
        
        f_plus = x_plus ** 2
        f_minus = x_minus ** 2
        
        numerical_grad = (f_plus - f_minus) / (2 * eps)
        
        # 解析梯度
        x_grad = torch.tensor([x_val], requires_grad=True)
        y = x_grad ** 2
        y.backward()
        
        analytical_grad = x_grad.grad.item()
        
        error = abs(numerical_grad.item() - analytical_grad)
        
        print(f"  数值梯度: {numerical_grad.item():.6f}")
        print(f"  解析梯度: {analytical_grad:.6f}")
        print(f"  误差: {error:.2e}")
    
    check_gradient(torch.tensor([2.0], requires_grad=True))


def main():
    """主函数"""
    print("="*60)
    print("  PyTorch自动微分示例")
    print("  Day 3: Python GPU编程 - PyTorch入门")
    print("="*60)
    
    # 演示各个主题
    basic_gradient()
    computation_graph()
    gradient_descent_simple()
    multi_variable_optimization()
    gradient_accumulation()
    learning_rate_effect()
    gpu_optimization()
    practical_tips()
    
    # 总结
    print_section("学习总结")
    print("\n✅ 你已经学习了:")
    print("  1. 梯度计算基础")
    print("  2. 计算图原理")
    print("  3. 梯度下降优化")
    print("  4. 多变量优化")
    print("  5. 梯度累加问题")
    print("  6. 学习率的影响")
    print("  7. GPU上的优化")
    print("  8. 实用技巧")
    
    print("\n🎯 关键要点:")
    print("  - 自动微分是深度学习的核心")
    print("  - 记得清零梯度,避免累加")
    print("  - 学习率是优化的关键超参数")
    print("  - GPU加速大规模优化问题")
    
    print("\n📚 下一步:")
    print("  - 运行 image_filter_gpu.py 学习GPU图像处理")
    print("  - 尝试修改优化参数,观察效果")
    print("  - 实现自己的优化问题")
    
    print("\n生成的图表:")
    print("  - optimization_1d.png: 一维优化过程")
    print("  - optimization_2d.png: 二维优化过程")
    print("  - learning_rate_comparison.png: 学习率对比")
    
    print("\n" + "="*60)
    print("练习完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
