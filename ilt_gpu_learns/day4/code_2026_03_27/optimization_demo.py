"""
优化算法演示程序

本程序演示优化算法在光刻技术中的应用,包括:
1. 梯度下降算法实现
2. 不同优化器比较
3. 优化路径可视化
4. 学习率影响分析

作者: GPU ILT学习体系
日期: 2026-03-27
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


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
        print(f"GPU型号: {gpu_name}")
    else:
        device = torch.device('cpu')
        print("未检测到GPU,将使用CPU运行")
    
    print()
    return device


def simple_gradient_descent(device):
    """
    最简单的梯度下降示例
    
    目标: 最小化 f(x) = x^2
    理论最优解: x = 0
    """
    print("="*60)
    print("1. 简单梯度下降示例")
    print("="*60)
    
    print("\n优化目标: 最小化 f(x) = x^2")
    print("理论最优解: x = 0")
    
    # 初始化
    x = torch.tensor([5.0], requires_grad=True, device=device)
    
    # 超参数
    learning_rate = 0.1
    iterations = 20
    
    print(f"\n参数设置:")
    print(f"  初始值: x = {x.item():.1f}")
    print(f"  学习率: {learning_rate}")
    print(f"  迭代次数: {iterations}")
    
    # 记录优化路径
    path_x = [x.item()]
    path_loss = [x.item() ** 2]
    
    print(f"\n优化过程:")
    for i in range(iterations):
        # 前向传播: 计算损失
        loss = x ** 2
        
        # 反向传播: 计算梯度
        loss.backward()
        
        # 更新参数
        with torch.no_grad():
            x -= learning_rate * x.grad
            
            # 清零梯度
            x.grad.zero_()
        
        # 记录
        path_x.append(x.item())
        path_loss.append(loss.item())
        
        if i % 5 == 0:
            print(f"  迭代 {i:2d}: x = {x.item():.6f}, loss = {loss.item():.6f}")
    
    print(f"\n最终结果:")
    print(f"  x = {x.item():.8f}")
    print(f"  loss = {path_loss[-1]:.8f}")
    print(f"  理论最优: x = 0.00000000, loss = 0.00000000")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 优化路径
    x_range = np.linspace(-6, 6, 100)
    y_range = x_range ** 2
    
    axes[0].plot(x_range, y_range, 'b-', label='f(x) = x²', alpha=0.5)
    axes[0].plot(path_x, path_loss, 'r.-', markersize=8, linewidth=2, label='优化路径')
    axes[0].plot(path_x[0], path_loss[0], 'go', markersize=12, label='起点')
    axes[0].plot(0, 0, 'r*', markersize=15, label='最优解')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Gradient Descent Path')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 损失曲线
    axes[1].semilogy(path_loss, 'g-', linewidth=2, marker='o', markersize=6)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Loss Curve')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_gradient_descent.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存为 simple_gradient_descent.png")
    
    return path_x, path_loss


def rosenbrock_optimization(device):
    """
    Rosenbrock函数优化
    
    经典的优化测试函数,考验优化算法性能
    目标: 最小化 f(x,y) = (a-x)^2 + b(y-x^2)^2
    理论最优解: (x, y) = (a, a^2) = (1, 1)
    """
    print("\n" + "="*60)
    print("2. Rosenbrock函数优化")
    print("="*60)
    
    def rosenbrock(x, y):
        """Rosenbrock函数"""
        a, b = 1.0, 100.0
        return (a - x)**2 + b * (y - x**2)**2
    
    print("\n优化目标: 最小化 Rosenbrock函数")
    print("理论最优解: (x, y) = (1, 1)")
    
    # 创建搜索空间(用于可视化)
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = rosenbrock(X, Y)
    
    # 初始化
    x = torch.tensor([-1.5], requires_grad=True, device=device)
    y = torch.tensor([2.5], requires_grad=True, device=device)
    
    # 超参数
    learning_rate = 0.002
    iterations = 2000
    
    print(f"\n参数设置:")
    print(f"  初始值: (x, y) = ({x.item():.1f}, {y.item():.1f})")
    print(f"  学习率: {learning_rate}")
    print(f"  迭代次数: {iterations}")
    
    # 记录优化路径
    path_x = [x.item()]
    path_y = [y.item()]
    losses = [rosenbrock(x.item(), y.item())]
    
    print(f"\n优化过程:")
    for i in range(iterations):
        # 计算损失
        loss = rosenbrock(x, y)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        with torch.no_grad():
            x -= learning_rate * x.grad
            y -= learning_rate * y.grad
            
            # 清零梯度
            x.grad.zero_()
            y.grad.zero_()
        
        # 记录
        path_x.append(x.item())
        path_y.append(y.item())
        losses.append(loss.item())
        
        if i % 500 == 0:
            print(f"  迭代 {i:4d}: x={x.item():.4f}, y={y.item():.4f}, loss={loss.item():.6f}")
    
    print(f"\n最终结果:")
    print(f"  位置: ({x.item():.6f}, {y.item():.6f})")
    print(f"  函数值: {losses[-1]:.6f}")
    print(f"  理论最优: (1.000000, 1.000000), 函数值=0")
    print(f"  误差: x误差={abs(x.item()-1):.6f}, y误差={abs(y.item()-1):.6f}")
    
    # 可视化
    fig = plt.figure(figsize=(15, 5))
    
    # 3D曲面
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, np.log(Z + 1), cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('log(f(x,y))')
    ax1.set_title('Rosenbrock Function\n(log scale)')
    
    # 等高线图
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
    ax2.plot(path_x, path_y, 'r.-', linewidth=1, markersize=2, label='优化路径')
    ax2.plot(path_x[0], path_y[0], 'go', markersize=10, label='起点')
    ax2.plot(1, 1, 'r*', markersize=15, label='最优解')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Optimization Path')
    ax2.legend()
    plt.colorbar(contour, ax=ax2, label='函数值')
    
    # 损失曲线
    ax3 = fig.add_subplot(133)
    ax3.semilogy(losses, 'b-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss (log scale)')
    ax3.set_title('Loss Curve')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rosenbrock_optimization.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存为 rosenbrock_optimization.png")
    
    return path_x, path_y, losses


def optimizer_comparison(device):
    """
    比较不同优化器的性能
    
    对比: SGD, Momentum, Adam
    """
    print("\n" + "="*60)
    print("3. 不同优化器比较")
    print("="*60)
    
    # 定义优化目标: 二次函数
    def quadratic(x):
        """简单的二次函数: f(x) = x^2"""
        return x ** 2
    
    # 不同优化算法的实现
    def gradient_descent(x_init, lr, iterations):
        """标准梯度下降"""
        x = torch.tensor([x_init], requires_grad=True, device=device)
        path = [x.item()]
        
        for _ in range(iterations):
            loss = quadratic(x)
            loss.backward()
            with torch.no_grad():
                x -= lr * x.grad
                x.grad.zero_()
            path.append(x.item())
        
        return path
    
    def momentum(x_init, lr, momentum_coeff, iterations):
        """带动量的梯度下降"""
        x = torch.tensor([x_init], requires_grad=True, device=device)
        velocity = torch.tensor([0.0], device=device)
        path = [x.item()]
        
        for _ in range(iterations):
            loss = quadratic(x)
            loss.backward()
            
            with torch.no_grad():
                velocity = momentum_coeff * velocity + lr * x.grad
                x -= velocity
                x.grad.zero_()
            path.append(x.item())
        
        return path
    
    def adam(x_init, lr, iterations):
        """Adam优化器"""
        x = torch.tensor([x_init], requires_grad=True, device=device)
        optimizer = torch.optim.Adam([x], lr=lr)
        path = [x.item()]
        
        for _ in range(iterations):
            optimizer.zero_grad()
            loss = quadratic(x)
            loss.backward()
            optimizer.step()
            path.append(x.item())
        
        return path
    
    # 测试参数
    x_init = 5.0
    iterations = 50
    lr = 0.1
    
    print(f"\n优化目标: 最小化 f(x) = x²")
    print(f"参数设置:")
    print(f"  初始值: x = {x_init}")
    print(f"  学习率: {lr}")
    print(f"  迭代次数: {iterations}")
    
    # 运行不同优化器
    path_gd = gradient_descent(x_init, lr, iterations)
    path_momentum = momentum(x_init, lr, 0.9, iterations)
    path_adam = adam(x_init, lr, iterations)
    
    print(f"\n优化结果:")
    print(f"  梯度下降: 最终x = {path_gd[-1]:.8f}")
    print(f"  动量法:   最终x = {path_momentum[-1]:.8f}")
    print(f"  Adam:     最终x = {path_adam[-1]:.8f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 优化路径
    x_range = np.linspace(-6, 6, 100)
    y_range = x_range ** 2
    
    axes[0].plot(x_range, y_range, 'b-', label='f(x) = x²', alpha=0.5)
    axes[0].plot(path_gd, [x**2 for x in path_gd], 'r.-', label='SGD', markersize=4)
    axes[0].plot(path_momentum, [x**2 for x in path_momentum], 'g.-', label='Momentum', markersize=4)
    axes[0].plot(path_adam, [x**2 for x in path_adam], 'm.-', label='Adam', markersize=4)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Optimization Path Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 收敛曲线
    axes[1].semilogy([abs(x) for x in path_gd], 'r-', label='SGD', linewidth=2)
    axes[1].semilogy([abs(x) for x in path_momentum], 'g-', label='Momentum', linewidth=2)
    axes[1].semilogy([abs(x) for x in path_adam], 'm-', label='Adam', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('|x| (log scale)')
    axes[1].set_title('Convergence Speed Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存为 optimizer_comparison.png")
    
    print(f"\n算法特点:")
    print(f"  梯度下降: 收敛稳定但较慢")
    print(f"  动量法: 收敛更快,能冲过局部最优")
    print(f"  Adam: 自适应学习率,适合复杂问题")
    
    return path_gd, path_momentum, path_adam


def learning_rate_analysis(device):
    """
    分析学习率对优化的影响
    """
    print("\n" + "="*60)
    print("4. 学习率影响分析")
    print("="*60)
    
    # 测试不同学习率
    learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
    iterations = 50
    x_init = 5.0
    
    print(f"\n优化目标: 最小化 f(x) = x²")
    print(f"初始值: x = {x_init}")
    print(f"测试学习率: {learning_rates}")
    
    paths = {}
    
    for lr in learning_rates:
        x = torch.tensor([x_init], requires_grad=True, device=device)
        path = [x.item()]
        
        for i in range(iterations):
            loss = x ** 2
            loss.backward()
            
            with torch.no_grad():
                x -= lr * x.grad
                x.grad.zero_()
            
            path.append(x.item())
        
        paths[lr] = path
        print(f"  学习率 {lr:.2f}: 最终x = {path[-1]:.8f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 优化路径
    x_range = np.linspace(-10, 10, 200)
    y_range = x_range ** 2
    
    axes[0].plot(x_range, y_range, 'b-', label='f(x) = x²', alpha=0.3, linewidth=2)
    
    for lr, path in paths.items():
        axes[0].plot(path, [x**2 for x in path], '.-', label=f'lr={lr}', markersize=4, linewidth=1.5)
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Effect of Learning Rate on Optimization Path')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-10, 10)
    
    # 收敛曲线
    for lr, path in paths.items():
        axes[1].semilogy([abs(x) for x in path], label=f'lr={lr}', linewidth=2)
    
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('|x| (log scale)')
    axes[1].set_title('Convergence Speed vs Learning Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存为 learning_rate_analysis.png")
    
    print(f"\n学习率选择建议:")
    print(f"  太小(lr=0.01): 收敛太慢")
    print(f"  适中(lr=0.05-0.1): 收敛速度和稳定性平衡")
    print(f"  太大(lr=0.5-1.0): 可能震荡甚至发散")
    
    return paths


def multivariable_optimization(device):
    """
    多变量优化示例
    
    目标: 最小化 f(x,y,z) = x^2 + y^2 + z^2
    理论最优解: (0, 0, 0)
    """
    print("\n" + "="*60)
    print("5. 多变量优化示例")
    print("="*60)
    
    print("\n优化目标: 最小化 f(x,y,z) = x² + y² + z²")
    print("理论最优解: (x, y, z) = (0, 0, 0)")
    
    # 初始化
    x = torch.tensor([3.0], requires_grad=True, device=device)
    y = torch.tensor([2.0], requires_grad=True, device=device)
    z = torch.tensor([1.0], requires_grad=True, device=device)
    
    # 使用Adam优化器
    optimizer = torch.optim.Adam([x, y, z], lr=0.5)
    
    iterations = 100
    
    print(f"\n参数设置:")
    print(f"  初始值: (x, y, z) = ({x.item()}, {y.item()}, {z.item()})")
    print(f"  优化器: Adam")
    print(f"  学习率: 0.5")
    print(f"  迭代次数: {iterations}")
    
    # 记录
    path = [[x.item(), y.item(), z.item()]]
    losses = [x.item()**2 + y.item()**2 + z.item()**2]
    
    print(f"\n优化过程:")
    for i in range(iterations):
        optimizer.zero_grad()
        
        loss = x**2 + y**2 + z**2
        loss.backward()
        
        optimizer.step()
        
        path.append([x.item(), y.item(), z.item()])
        losses.append(loss.item())
        
        if i % 20 == 0:
            print(f"  迭代 {i:3d}: x={x.item():.6f}, y={y.item():.6f}, z={z.item():.6f}, loss={loss.item():.8f}")
    
    print(f"\n最终结果:")
    print(f"  位置: ({x.item():.8f}, {y.item():.8f}, {z.item():.8f})")
    print(f"  函数值: {losses[-1]:.8f}")
    print(f"  理论最优: (0, 0, 0), 函数值=0")
    
    # 可视化
    fig = plt.figure(figsize=(12, 5))
    
    # 3D优化路径
    ax1 = fig.add_subplot(121, projection='3d')
    path_array = np.array(path)
    ax1.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'r.-', markersize=2)
    ax1.plot([path_array[0, 0]], [path_array[0, 1]], [path_array[0, 2]], 'go', markersize=10, label='起点')
    ax1.plot([0], [0], [0], 'r*', markersize=15, label='最优解')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Optimization Path in 3D')
    ax1.legend()
    
    # 损失曲线
    ax2 = fig.add_subplot(122)
    ax2.semilogy(losses, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Loss Curve')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multivariable_optimization.png', dpi=150, bbox_inches='tight')
    print(f"\n可视化已保存为 multivariable_optimization.png")
    
    return path, losses


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Day 4: 优化算法演示程序")
    print("="*60)
    
    # 检查GPU
    device = check_gpu()
    
    # 1. 简单梯度下降
    path_x, path_loss = simple_gradient_descent(device)
    
    # 2. Rosenbrock函数优化
    path_x, path_y, losses = rosenbrock_optimization(device)
    
    # 3. 优化器比较
    path_gd, path_momentum, path_adam = optimizer_comparison(device)
    
    # 4. 学习率分析
    paths = learning_rate_analysis(device)
    
    # 5. 多变量优化
    path, losses = multivariable_optimization(device)
    
    print("\n" + "="*60)
    print("所有演示完成!")
    print("="*60)
    print("\n生成的文件:")
    print("  1. simple_gradient_descent.png - 简单梯度下降")
    print("  2. rosenbrock_optimization.png - Rosenbrock函数优化")
    print("  3. optimizer_comparison.png - 优化器比较")
    print("  4. learning_rate_analysis.png - 学习率分析")
    print("  5. multivariable_optimization.png - 多变量优化")
    print("\n请查看这些图像以加深理解!")


if __name__ == "__main__":
    main()
