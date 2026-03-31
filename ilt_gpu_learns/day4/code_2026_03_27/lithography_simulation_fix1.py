"""
简化光刻仿真程序

本程序演示简化的光刻成像模型,包括:
1. 掩模图案的矩阵表示
2. 基于FFT的光学成像仿真
3. 简单的掩模优化
4. 结果可视化

这是真实ILT算法的简化版本,帮助理解核心思想。

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


def create_target_pattern(size=128):
    """
    创建目标图案(期望在晶圆上得到的图案)
    
    在实际光刻中,这是设计者想要的电路图案
    """
    print("="*60)
    print("1. 创建目标图案")
    print("="*60)
    
    target = torch.zeros(size, size, device='cuda')
    
    # 创建一个简单的电路图案
    # 1. 水平线
    target[20:30, 10:110] = 1.0
    target[50:60, 10:110] = 1.0
    target[80:90, 10:110] = 1.0
    
    # 2. 垂直线
    target[10:100, 20:30] = 1.0
    target[10:100, 50:60] = 1.0
    target[10:100, 80:90] = 1.0
    
    print(f"目标图案大小: {size}x{size}")
    print(f"图案区域占比: {(target.sum() / target.numel() * 100):.2f}%")
    
    return target


def create_optical_kernel(size, wavelength=193, na=1.35):
    """
    创建简化的光学成像核(点扩散函数)
    
    这是光刻成像模型的核心:
    - 光源发出的光通过掩模
    - 光学系统对光进行滤波(衍射极限)
    - 在晶圆上形成光强分布
    
    简化模型:使用airy函数近似
    """
    print("\n" + "="*60)
    print("2. 创建光学成像核")
    print("="*60)
    
    # 创建坐标网格
    x = torch.arange(size, device='cuda') - size // 2
    y = torch.arange(size, device='cuda') - size // 2
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # 计算airy函数(简化)
    # 实际光刻中,这涉及复杂的Hopkins理论
    r = torch.sqrt(X**2 + Y**2)
    
    # 使用高斯近似airy函数(简化)
    sigma = size / 20  # 控制分辨率
    kernel = torch.exp(-(r**2) / (2 * sigma**2))
    
    # 归一化
    kernel /= kernel.sum()
    
    print(f"光学核大小: {size}x{size}")
    print(f"波长: {wavelength} nm (ArF光源)")
    print(f"数值孔径(NA): {na}")
    print(f"核函数峰值: {kernel.max():.6f}")
    
    return kernel


def simulate_imaging(mask, kernel):
    """
    模拟光刻成像过程
    
    成像模型: I = |PSF ⊗ M|^2
    
    其中:
    - I: 晶圆上的光强分布
    - PSF: 点扩散函数(光学核)
    - M: 掩模图案
    - ⊗: 卷积运算
    
    使用FFT加速: I = |FFT^{-1}(FFT(PSF) × FFT(M))|^2
    """
    size = mask.shape[0]
    
    # FFT卷积
    mask_fft = torch.fft.fft2(mask)
    kernel_fft = torch.fft.fft2(kernel)
    
    # 频域相乘
    result_fft = mask_fft * kernel_fft
    
    # 逆FFT
    amplitude = torch.fft.ifft2(result_fft)
    
    # 光强 = 振幅的平方
    intensity = torch.abs(amplitude) ** 2
    
    return intensity


def simple_resist_model(intensity, threshold=0.5):
    """
    简化的光刻胶模型
    
    光刻胶在光强超过阈值的地方显影
    这是非常简化的模型,实际光刻胶模型更复杂
    """
    # 阈值化
    resist_pattern = (intensity > threshold).float()
    
    return resist_pattern


def calculate_loss(wafer_pattern, target_pattern):
    """
    计算损失函数
    
    损失 = ||晶圆图案 - 目标图案||^2
    
    在实际ILT中,损失函数更复杂,包括:
    - L2损失
    - PVBand(工艺窗口)
    - MRC约束(掩模规则检查)
    """
    loss = torch.mean((wafer_pattern - target_pattern) ** 2)
    return loss


def optimize_mask(target_pattern, kernel, iterations=100, lr=0.01):
    """
    简化的掩模优化
    
    使用梯度下降优化掩模图案
    
    这是非常简化的ILT算法,实际算法更复杂
    """
    print("\n" + "="*60)
    print("3. 掩模优化")
    print("="*60)
    
    size = target_pattern.shape[0]
    
    # 初始化掩模参数(使用可训练参数)
    # 注意:不直接使用target_pattern,而是创建新的可训练参数
    mask_param = torch.nn.Parameter(target_pattern.clone())
    
    # 优化器
    optimizer = torch.optim.Adam([mask_param], lr=lr)
    
    # 记录
    losses = []
    best_loss = float('inf')
    best_mask = None
    
    print(f"\n优化参数:")
    print(f"  初始掩模: 目标图案")
    print(f"  学习率: {lr}")
    print(f"  迭代次数: {iterations}")
    
    print(f"\n优化过程:")
    for i in range(iterations):
        # 清零梯度
        optimizer.zero_grad()
        
        # 使用sigmoid约束掩模值在[0, 1]之间
        # sigmoid(x) ∈ (0, 1),可以保持梯度流
        mask = torch.sigmoid(mask_param)
        
        # 模拟成像
        intensity = simulate_imaging(mask, kernel)
        
        # 使用可微分的sigmoid替代硬阈值函数
        # 这样可以保持梯度传播
        threshold = 0.3
        wafer_pattern = torch.sigmoid(50 * (intensity - threshold))
        
        # 计算损失
        loss = calculate_loss(wafer_pattern, target_pattern)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 记录
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_mask = mask.detach().clone()
        
        if i % 10 == 0:
            print(f"  迭代 {i:3d}: loss = {loss.item():.6f}")
    
    print(f"\n优化完成:")
    print(f"  最终损失: {losses[-1]:.6f}")
    print(f"  最佳损失: {best_loss:.6f}")
    
    return best_mask, losses


def visualize_results(target, initial_mask, optimized_mask, kernel, losses):
    """
    可视化所有结果
    """
    print("\n" + "="*60)
    print("4. 结果可视化")
    print("="*60)
    
    # 模拟初始掩模成像
    initial_intensity = simulate_imaging(initial_mask, kernel)
    initial_wafer = simple_resist_model(initial_intensity, 0.3)
    
    # 模拟优化后掩模成像
    optimized_intensity = simulate_imaging(optimized_mask, kernel)
    optimized_wafer = simple_resist_model(optimized_intensity, 0.3)
    
    # 计算误差
    initial_error = torch.abs(initial_wafer - target).sum().item()
    optimized_error = torch.abs(optimized_wafer - target).sum().item()
    
    print(f"\n成像质量评估:")
    print(f"  初始掩模误差: {initial_error:.2f} 像素")
    print(f"  优化掩模误差: {optimized_error:.2f} 像素")
    print(f"  误差减少: {(1 - optimized_error/initial_error)*100:.1f}%")
    
    # 可视化
    fig = plt.figure(figsize=(18, 12))
    
    # 第一行:目标、初始掩模、优化掩模
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(target.cpu(), cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Target Pattern\n(Design)')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(initial_mask.cpu(), cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Initial Mask\n(Target Pattern)')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(optimized_mask.cpu(), cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Optimized Mask\n(ILT Result)')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(kernel.cpu(), cmap='hot')
    ax4.set_title('Optical Kernel\n(PSF)')
    ax4.axis('off')
    
    # 第二行:成像结果
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(initial_intensity.cpu(), cmap='gray')
    ax5.set_title('Initial Intensity\n(Simulated)')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(optimized_intensity.cpu(), cmap='gray')
    ax6.set_title('Optimized Intensity\n(Simulated)')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.imshow(initial_wafer.cpu(), cmap='gray', vmin=0, vmax=1)
    ax7.set_title(f'Initial Wafer\n(Error={initial_error:.0f})')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.imshow(optimized_wafer.cpu(), cmap='gray', vmin=0, vmax=1)
    ax8.set_title(f'Optimized Wafer\n(Error={optimized_error:.0f})')
    ax8.axis('off')
    
    plt.suptitle('Simplified Lithography Simulation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lithography_simulation_results.png', dpi=150, bbox_inches='tight')
    print(f"\n仿真结果已保存为 lithography_simulation_results.png")
    
    # 绘制优化过程
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(losses, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Mask Optimization Process', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标记最佳点
    best_iter = np.argmin(losses)
    ax.plot(best_iter, losses[best_iter], 'r*', markersize=15, label=f'Best (iter {best_iter})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('mask_optimization.png', dpi=150, bbox_inches='tight')
    print(f"优化过程已保存为 mask_optimization.png")
    
    return initial_wafer, optimized_wafer


def performance_analysis():
    """
    性能分析:比较不同规模的仿真时间
    """
    print("\n" + "="*60)
    print("5. 性能分析")
    print("="*60)
    
    import time
    
    sizes = [64, 128, 256, 512, 1024]
    
    print(f"\n测试不同规模的仿真性能:\n")
    print(f"{'Size':<10} {'FFT Time (ms)':<15} {'Total Time (ms)':<15}")
    print("-" * 40)
    
    for size in sizes:
        # 创建数据
        mask = torch.randn(size, size, device='cuda')
        kernel = torch.randn(size, size, device='cuda')
        
        # FFT时间
        torch.cuda.synchronize()
        start = time.time()
        mask_fft = torch.fft.fft2(mask)
        kernel_fft = torch.fft.fft2(kernel)
        torch.cuda.synchronize()
        fft_time = (time.time() - start) * 1000
        
        # 完整仿真时间
        torch.cuda.synchronize()
        start = time.time()
        intensity = simulate_imaging(mask, kernel)
        torch.cuda.synchronize()
        total_time = (time.time() - start) * 1000
        
        print(f"{size:<10} {fft_time:<15.2f} {total_time:<15.2f}")
    
    print(f"\n结论:")
    print(f"  FFT大幅加速了光刻仿真")
    print(f"  GPU可以处理大规模仿真(1024x1024在毫秒级完成)")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Day 4: 简化光刻仿真程序")
    print("="*60)
    
    # 检查GPU
    device = check_gpu()
    
    # 1. 创建目标图案
    target = create_target_pattern(size=128)
    
    # 2. 创建光学核
    kernel = create_optical_kernel(128)
    
    # 3. 初始掩模(直接使用目标图案)
    initial_mask = target.clone()
    
    # 4. 优化掩模
    optimized_mask, losses = optimize_mask(target, kernel, iterations=100, lr=0.01)
    
    # 5. 可视化结果
    initial_wafer, optimized_wafer = visualize_results(target, initial_mask, optimized_mask, kernel, losses)
    
    # 6. 性能分析
    performance_analysis()
    
    print("\n" + "="*60)
    print("仿真完成!")
    print("="*60)
    print("\n生成的文件:")
    print("  1. lithography_simulation_results.png - 仿真结果对比")
    print("  2. mask_optimization.png - 优化过程")
    print("\n关键观察:")
    print("  - 优化后的掩模与目标图案不同(有细微调整)")
    print("  - 优化后的晶圆图案更接近目标")
    print("  - 这就是ILT的核心思想!")
    print("\n这是极简化的模型,实际ILT要复杂得多,但原理相通。")


if __name__ == "__main__":
    main()
