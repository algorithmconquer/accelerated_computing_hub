"""
傅里叶变换分析程序

本程序演示傅里叶变换在光刻技术中的应用,包括:
1. 一维信号的FFT分析
2. 二维图像的频谱分析
3. 频域滤波实现
4. 卷积定理验证

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


def fft_1d_analysis(device):
    """
    一维信号的傅里叶变换分析
    
    应用场景:
    - 分析光源的时间特性
    - 振动分析
    - 信号处理基础
    """
    print("="*60)
    print("1. 一维信号的FFT分析")
    print("="*60)
    
    # 创建一个复合信号
    fs = 1000  # 采样频率
    t = torch.linspace(0, 1, fs, device=device)  # 1秒,1000个采样点
    
    # 信号 = 50Hz正弦波 + 120Hz正弦波 + 噪声
    f1, f2 = 50, 120
    signal_clean = torch.sin(2 * np.pi * f1 * t) + 0.5 * torch.sin(2 * np.pi * f2 * t)
    noise = 0.2 * torch.randn_like(t)
    signal = signal_clean + noise
    
    print(f"创建复合信号:")
    print(f"  采样频率: {fs} Hz")
    print(f"  信号成分: {f1}Hz正弦波(振幅1.0) + {f2}Hz正弦波(振幅0.5)")
    print(f"  噪声: 高斯白噪声(振幅0.2)")
    
    # 应用FFT
    signal_fft = torch.fft.fft(signal)
    
    # 计算频率轴
    freqs = torch.fft.fftfreq(fs, 1/fs, device=device)
    
    # 计算幅度谱
    magnitude = torch.abs(signal_fft)
    magnitude_db = 20 * torch.log10(magnitude + 1e-10)
    
    print(f"\nFFT分析:")
    print(f"  FFT输出形状: {signal_fft.shape}")
    print(f"  频率范围: {freqs.min():.0f} Hz 到 {freqs.max():.0f} Hz")
    
    # 找到主要频率分量
    positive_freq_mask = freqs > 0
    positive_freqs = freqs[positive_freq_mask]
    positive_magnitude = magnitude[positive_freq_mask]
    
    # 找到前3个峰值
    top3_indices = torch.topk(positive_magnitude, 3).indices
    print(f"\n检测到的主要频率分量:")
    for idx in top3_indices:
        freq = positive_freqs[idx].item()
        mag = positive_magnitude[idx].item()
        print(f"  频率: {freq:.1f} Hz, 幅度: {mag:.2f}")
    
    # 可视化
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 时间域信号(全部)
    axes[0].plot(t.cpu(), signal.cpu())
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Domain Signal (Full)')
    axes[0].grid(True, alpha=0.3)
    
    # 时间域信号(前100个采样点)
    axes[1].plot(t[:100].cpu(), signal[:100].cpu())
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Time Domain Signal (First 100 samples)')
    axes[1].grid(True, alpha=0.3)
    
    # 频率域信号
    axes[2].plot(positive_freqs.cpu(), positive_magnitude.cpu())
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude')
    axes[2].set_title('Frequency Domain (Magnitude Spectrum)')
    axes[2].set_xlim(0, 200)
    axes[2].grid(True, alpha=0.3)
    
    # 标记检测到的峰值
    for idx in top3_indices[:2]:  # 只标记前2个
        freq = positive_freqs[idx].item()
        mag = positive_magnitude[idx].item()
        axes[2].axvline(x=freq, color='r', linestyle='--', alpha=0.5)
        axes[2].plot(freq, mag, 'ro', markersize=10)
    
    plt.tight_layout()
    plt.savefig('fft_1d_example.png', dpi=150, bbox_inches='tight')
    print(f"\n一维FFT分析可视化已保存为 fft_1d_example.png")
    
    return signal, signal_fft


def fft_2d_analysis(device):
    """
    二维图像的傅里叶变换分析
    
    在光刻中的应用:
    - 掩模图案的频谱分析
    - 光学系统的频率响应
    - 衍射图样计算
    """
    print("\n" + "="*60)
    print("2. 二维图像的FFT分析")
    print("="*60)
    
    # 创建一个测试掩模图案
    size = 128
    mask = torch.zeros(size, size, device=device)
    
    # 添加几何图形
    # 1. 矩形孔径
    mask[40:60, 30:50] = 1.0
    
    # 2. 圆形孔径
    center = (80, 80)
    radius = 15
    y, x = torch.meshgrid(torch.arange(size, device=device), 
                           torch.arange(size, device=device), indexing='ij')
    circle_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    mask[circle_mask] = 1.0
    
    # 3. 细线条(高频成分)
    mask[100:105, 70:120] = 1.0
    
    print(f"创建掩模图案: {size}x{size}")
    print(f"透光区域占比: {(mask.sum() / mask.numel() * 100):.2f}%")
    
    # 二维傅里叶变换
    mask_fft = torch.fft.fft2(mask)
    mask_fft_shifted = torch.fft.fftshift(mask_fft)  # 将零频率移到中心
    
    # 计算幅度谱
    magnitude = torch.abs(mask_fft_shifted)
    magnitude_log = torch.log(magnitude + 1)  # 对数尺度
    
    # 计算相位谱
    phase = torch.angle(mask_fft_shifted)
    
    print(f"\n二维FFT分析:")
    print(f"  FFT输出形状: {mask_fft.shape}")
    print(f"  幅度谱范围: {magnitude.min():.2f} 到 {magnitude.max():.2f}")
    
    # 频域分析
    print(f"\n频域分析:")
    print(f"  - 中心亮点: 直流分量(图像平均值)")
    print(f"  - 水平/垂直线: 对应边缘和线条")
    print(f"  - 中心周围的光晕: 高频成分(细节)")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始掩模
    axes[0, 0].imshow(mask.cpu(), cmap='gray')
    axes[0, 0].set_title('Mask Pattern (Spatial Domain)')
    axes[0, 0].axis('off')
    
    # 频谱(线性尺度)
    im1 = axes[0, 1].imshow(magnitude.cpu(), cmap='hot')
    axes[0, 1].set_title('Magnitude Spectrum (Linear)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 频谱(对数尺度)
    im2 = axes[0, 2].imshow(magnitude_log.cpu(), cmap='hot')
    axes[0, 2].set_title('Magnitude Spectrum (Log Scale)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 相位谱
    im3 = axes[1, 0].imshow(phase.cpu(), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[1, 0].set_title('Phase Spectrum')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 频谱的3D视图(中心区域)
    center_size = 32
    center_start = size // 2 - center_size // 2
    center_end = size // 2 + center_size // 2
    
    from mpl_toolkits.mplot3d import Axes3D
    ax3d = fig.add_subplot(2, 3, 5, projection='3d')
    
    x_3d = np.arange(center_size)
    y_3d = np.arange(center_size)
    X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
    
    center_magnitude = magnitude_log[center_start:center_end, center_start:center_end].cpu()
    
    ax3d.plot_surface(X_3d, Y_3d, center_magnitude, cmap='hot', alpha=0.8)
    ax3d.set_xlabel('Frequency X')
    ax3d.set_ylabel('Frequency Y')
    ax3d.set_zlabel('Log Magnitude')
    ax3d.set_title('3D View of Center Region')
    
    # 逆FFT重建
    mask_reconstructed = torch.fft.ifft2(mask_fft).real
    
    axes[1, 2].imshow(mask_reconstructed.cpu(), cmap='gray')
    axes[1, 2].set_title('Reconstructed Mask (IFFT)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('fft_2d_mask.png', dpi=150, bbox_inches='tight')
    print(f"\n二维FFT分析可视化已保存为 fft_2d_mask.png")
    
    # 验证重建
    reconstruction_error = torch.abs(mask - mask_reconstructed).max()
    print(f"\n重建误差: {reconstruction_error:.8f} (应接近0)")
    
    return mask, mask_fft


def frequency_domain_filtering(device):
    """
    频域滤波演示
    
    在光刻中的应用:
    - 低通滤波: 去除高频噪声
    - 高通滤波: 边缘检测
    - 带通滤波: 提取特定频率成分
    """
    print("\n" + "="*60)
    print("3. 频域滤波")
    print("="*60)
    
    # 创建一个带有噪声的图像
    size = 128
    image_clean = torch.zeros(size, size, device=device)
    image_clean[40:80, 30:90] = 1.0
    
    # 添加高频噪声
    noise = 0.3 * torch.randn(size, size, device=device)
    image_noisy = image_clean + noise
    image_noisy = torch.clamp(image_noisy, 0, 1)
    
    print(f"创建带噪声的图像: {size}x{size}")
    
    # FFT
    image_fft = torch.fft.fft2(image_noisy)
    image_fft_shifted = torch.fft.fftshift(image_fft)
    
    # 创建低通滤波器
    y, x = torch.meshgrid(torch.arange(size, device=device) - size//2,
                           torch.arange(size, device=device) - size//2, indexing='ij')
    radius = size // 4
    lowpass_filter = (x**2 + y**2 <= radius**2).float()
    
    # 创建高通滤波器
    highpass_filter = (x**2 + y**2 > radius**2).float()
    
    print(f"\n创建滤波器:")
    print(f"  低通滤波器: 半径={radius}, 保留低频成分")
    print(f"  高通滤波器: 半径={radius}, 保留高频成分")
    
    # 应用低通滤波
    lowpassed_fft = image_fft_shifted * lowpass_filter
    lowpassed_image = torch.fft.ifft2(torch.fft.ifftshift(lowpassed_fft)).real
    
    # 应用高通滤波
    highpassed_fft = image_fft_shifted * highpass_filter
    highpassed_image = torch.fft.ifft2(torch.fft.ifftshift(highpassed_fft)).real
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 原始图像
    axes[0, 0].imshow(image_clean.cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 加噪图像
    axes[0, 1].imshow(image_noisy.cpu(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')
    
    # 低通滤波器
    axes[0, 2].imshow(lowpass_filter.cpu(), cmap='gray')
    axes[0, 2].set_title('Lowpass Filter')
    axes[0, 2].axis('off')
    
    # 高通滤波器
    axes[0, 3].imshow(highpass_filter.cpu(), cmap='gray')
    axes[0, 3].set_title('Highpass Filter')
    axes[0, 3].axis('off')
    
    # 频谱
    axes[1, 0].imshow(torch.log(torch.abs(image_fft_shifted) + 1).cpu(), cmap='hot')
    axes[1, 0].set_title('Original Spectrum')
    axes[1, 0].axis('off')
    
    # 低通滤波后频谱
    axes[1, 1].imshow(torch.log(torch.abs(lowpassed_fft) + 1).cpu(), cmap='hot')
    axes[1, 1].set_title('Lowpassed Spectrum')
    axes[1, 1].axis('off')
    
    # 低通滤波后图像
    axes[1, 2].imshow(lowpassed_image.cpu(), cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Lowpassed Image\n(Noise Removed)')
    axes[1, 2].axis('off')
    
    # 高通滤波后图像
    axes[1, 3].imshow(torch.abs(highpassed_image).cpu(), cmap='gray')
    axes[1, 3].set_title('Highpassed Image\n(Edges Detected)')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('frequency_filtering.png', dpi=150, bbox_inches='tight')
    print(f"\n频域滤波可视化已保存为 frequency_filtering.png")
    
    print(f"\n滤波效果:")
    print(f"  低通滤波: 去除噪声,图像更平滑")
    print(f"  高通滤波: 提取边缘,突出细节")
    
    return image_noisy, lowpassed_image, highpassed_image


def convolution_theorem_verification(device):
    """
    验证卷积定理: 空间域卷积 = 频域乘法
    
    这是光刻成像仿真的理论基础!
    """
    print("\n" + "="*60)
    print("4. 卷积定理验证")
    print("="*60)
    
    import time
    
    # 创建大图像和卷积核
    image_size = 1024
    kernel_size = 31
    
    # 创建测试图像
    image = torch.randn(image_size, image_size, device=device)
    
    # 创建高斯卷积核
    x = torch.arange(kernel_size, device=device) - kernel_size // 2
    gaussian_1d = torch.exp(-x**2 / (2 * 5.0**2))
    gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
    gaussian_2d /= gaussian_2d.sum()  # 归一化
    
    print(f"测试卷积定理:")
    print(f"  图像大小: {image_size}x{image_size}")
    print(f"  卷积核大小: {kernel_size}x{kernel_size}")
    
    # 方法1: 空间域卷积
    from torch.nn.functional import conv2d
    
    start = time.time()
    image_4d = image.unsqueeze(0).unsqueeze(0)
    kernel_4d = gaussian_2d.unsqueeze(0).unsqueeze(0)
    result_spatial = conv2d(image_4d, kernel_4d, padding=kernel_size//2)
    torch.cuda.synchronize()
    time_spatial = time.time() - start
    
    print(f"\n方法1: 空间域卷积")
    print(f"  耗时: {time_spatial*1000:.2f} ms")
    
    # 方法2: 频域卷积
    start = time.time()
    
    # 计算填充后的大小
    padded_size = image_size + kernel_size - 1
    
    # 填充图像和核
    image_padded = torch.zeros(padded_size, padded_size, device=device)
    image_padded[:image_size, :image_size] = image
    
    kernel_padded = torch.zeros(padded_size, padded_size, device=device)
    kernel_padded[:kernel_size, :kernel_size] = gaussian_2d
    
    # FFT
    image_fft = torch.fft.fft2(image_padded)
    kernel_fft = torch.fft.fft2(kernel_padded)
    
    # 频域相乘
    result_fft = image_fft * kernel_fft
    
    # 逆FFT
    result_frequency_full = torch.fft.ifft2(result_fft).real
    
    # 提取有效区域
    result_frequency = result_frequency_full[kernel_size//2:kernel_size//2+image_size, 
                                              kernel_size//2:kernel_size//2+image_size]
    
    torch.cuda.synchronize()
    time_frequency = time.time() - start
    
    print(f"\n方法2: 频域卷积")
    print(f"  耗时: {time_frequency*1000:.2f} ms")
    
    # 比较结果
    difference = torch.abs(result_spatial.squeeze() - result_frequency).max()
    
    print(f"\n结果比较:")
    print(f"  最大差异: {difference:.8f}")
    print(f"  (差异应接近0,验证卷积定理)")
    
    print(f"\n性能比较:")
    print(f"  空间域卷积: {time_spatial*1000:.2f} ms")
    print(f"  频域卷积: {time_frequency*1000:.2f} ms")
    
    if time_spatial < time_frequency:
        print(f"  结论: 对于小卷积核({kernel_size}x{kernel_size}),空间域方法更快")
    else:
        speedup = time_spatial / time_frequency
        print(f"  结论: 对于大卷积核,频域方法更快 (加速比: {speedup:.2f}x)")
    
    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 原始图像
    axes[0].imshow(image.cpu(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 卷积核
    axes[1].imshow(gaussian_2d.cpu(), cmap='hot')
    axes[1].set_title('Gaussian Kernel')
    axes[1].axis('off')
    
    # 空间域结果
    axes[2].imshow(result_spatial.squeeze().cpu(), cmap='gray')
    axes[2].set_title(f'Spatial Convolution\n({time_spatial*1000:.1f} ms)')
    axes[2].axis('off')
    
    # 频域结果
    axes[3].imshow(result_frequency.cpu(), cmap='gray')
    axes[3].set_title(f'Frequency Convolution\n({time_frequency*1000:.1f} ms)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('convolution_theorem.png', dpi=150, bbox_inches='tight')
    print(f"\n卷积定理验证可视化已保存为 convolution_theorem.png")
    
    return result_spatial, result_frequency


def fft_performance_analysis(device):
    """
    FFT性能分析
    """
    print("\n" + "="*60)
    print("5. FFT性能分析")
    print("="*60)
    
    import time
    
    sizes = [64, 128, 256, 512, 1024, 2048]
    
    print(f"\n测试不同大小的FFT性能:\n")
    print(f"{'Size':<10} {'FFT Time (ms)':<15} {'IFFT Time (ms)':<15}")
    print("-" * 40)
    
    results = []
    for size in sizes:
        # 创建随机复数矩阵
        data = torch.randn(size, size, device=device)
        
        # FFT
        torch.cuda.synchronize()
        start = time.time()
        fft_result = torch.fft.fft2(data)
        torch.cuda.synchronize()
        fft_time = (time.time() - start) * 1000
        
        # IFFT
        torch.cuda.synchronize()
        start = time.time()
        ifft_result = torch.fft.ifft2(fft_result)
        torch.cuda.synchronize()
        ifft_time = (time.time() - start) * 1000
        
        print(f"{size:<10} {fft_time:<15.2f} {ifft_time:<15.2f}")
        
        results.append({
            'size': size,
            'fft_time': fft_time,
            'ifft_time': ifft_time
        })
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sizes_list = [r['size'] for r in results]
    fft_times = [r['fft_time'] for r in results]
    ifft_times = [r['ifft_time'] for r in results]
    
    x = np.arange(len(sizes_list))
    width = 0.35
    
    ax.bar(x - width/2, fft_times, width, label='FFT', alpha=0.8)
    ax.bar(x + width/2, ifft_times, width, label='IFFT', alpha=0.8)
    
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('FFT Performance Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}×{s}' for s in sizes_list])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('fft_performance.png', dpi=150, bbox_inches='tight')
    print(f"\nFFT性能分析已保存为 fft_performance.png")
    
    return results


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Day 4: 傅里叶变换分析程序")
    print("="*60)
    
    # 检查GPU
    device = check_gpu()
    
    # 1. 一维FFT分析
    signal, signal_fft = fft_1d_analysis(device)
    
    # 2. 二维FFT分析
    mask, mask_fft = fft_2d_analysis(device)
    
    # 3. 频域滤波
    image_noisy, lowpassed, highpassed = frequency_domain_filtering(device)
    
    # 4. 卷积定理验证
    result_spatial, result_frequency = convolution_theorem_verification(device)
    
    # 5. FFT性能分析
    results = fft_performance_analysis(device)
    
    print("\n" + "="*60)
    print("所有演示完成!")
    print("="*60)
    print("\n生成的文件:")
    print("  1. fft_1d_example.png - 一维FFT分析")
    print("  2. fft_2d_mask.png - 二维FFT分析")
    print("  3. frequency_filtering.png - 频域滤波")
    print("  4. convolution_theorem.png - 卷积定理验证")
    print("  5. fft_performance.png - FFT性能分析")
    print("\n请查看这些图像以加深理解!")


if __name__ == "__main__":
    main()
