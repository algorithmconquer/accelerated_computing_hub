#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3: GPU加速图像滤波器
=======================

本脚本实现GPU加速的图像处理,包括:
1. 图像卷积基础
2. 多种卷积核效果
3. CPU与GPU性能对比
4. 实时滤波演示

图像卷积是光刻成像仿真的核心技术之一。
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


def print_section(title):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def create_test_image(size=256):
    """创建测试图像:包含几何图形"""
    image = np.zeros((size, size), dtype=np.float32)
    
    # 添加矩形
    image[50:100, 50:150] = 1.0
    
    # 添加圆形
    center = (180, 180)
    radius = 40
    y, x = np.ogrid[:size, :size]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    image[mask] = 1.0
    
    # 添加三角形
    for i in range(50):
        image[150+i, 20+i*2:40+i*2] = 1.0
    
    # 添加渐变背景
    gradient = np.linspace(0, 0.3, size)
    image += gradient.reshape(-1, 1)
    
    # 添加噪声
    noise = np.random.randn(size, size) * 0.05
    image += noise
    
    # 归一化到[0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def get_convolution_kernels():
    """定义常用卷积核"""
    kernels = {}
    
    # 1. 均值滤波(模糊)
    kernels['均值滤波(3x3)'] = torch.ones(1, 1, 3, 3) / 9.0
    
    # 2. 高斯模糊
    gaussian = torch.tensor([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3) / 16.0
    kernels['高斯模糊(3x3)'] = gaussian
    
    # 3. 边缘检测(Sobel算子)
    sobel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    kernels['Sobel边缘(X)'] = sobel_x
    kernels['Sobel边缘(Y)'] = sobel_y
    
    # 4. Laplacian边缘检测
    laplacian = torch.tensor([[ 0, -1,  0],
                               [-1,  4, -1],
                               [ 0, -1,  0]], dtype=torch.float32).reshape(1, 1, 3, 3)
    kernels['Laplacian边缘'] = laplacian
    
    # 5. 锐化
    sharpen = torch.tensor([[ 0, -1,  0],
                             [-1,  5, -1],
                             [ 0, -1,  0]], dtype=torch.float32).reshape(1, 1, 3, 3)
    kernels['锐化'] = sharpen
    
    # 6. 浮雕效果
    emboss = torch.tensor([[-2, -1, 0],
                            [-1,  1, 1],
                            [ 0,  1, 2]], dtype=torch.float32).reshape(1, 1, 3, 3)
    kernels['浮雕效果'] = emboss
    
    # 7. Prewitt边缘检测
    prewitt_x = torch.tensor([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    prewitt_y = torch.tensor([[-1, -1, -1],
                               [ 0,  0,  0],
                               [ 1,  1,  1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    kernels['Prewitt边缘(X)'] = prewitt_x
    kernels['Prewitt边缘(Y)'] = prewitt_y
    
    return kernels


def apply_convolution_cpu(image, kernel):
    """CPU卷积实现"""
    # 转换为PyTorch张量
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    kernel_tensor = kernel.float()
    
    # 应用卷积
    result = F.conv2d(image_tensor, kernel_tensor, padding=1)
    
    return result.squeeze().numpy()


def apply_convolution_gpu(image, kernel, device='cuda'):
    """GPU卷积实现"""
    # 转换为PyTorch张量并移到GPU
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    kernel_tensor = kernel.float().to(device)
    
    # 应用卷积
    result = F.conv2d(image_tensor, kernel_tensor, padding=1)
    
    # 移回CPU
    return result.cpu().squeeze().numpy()


def compare_performance(image, kernels, device):
    """对比CPU和GPU性能"""
    print_section("性能对比测试")
    
    sizes = [256, 512, 1024, 2048]
    
    print(f"\n{'图像尺寸':<12} {'CPU时间':<12} {'GPU时间':<12} {'加速比':<10}")
    print("-" * 50)
    
    for size in sizes:
        # 创建不同尺寸的图像
        test_image = create_test_image(size)
        
        # CPU测试
        kernel = kernels['高斯模糊(3x3)']
        
        start = time.time()
        result_cpu = apply_convolution_cpu(test_image, kernel)
        cpu_time = time.time() - start
        
        # GPU测试
        if torch.cuda.is_available():
            # 预热
            _ = apply_convolution_gpu(test_image, kernel, device)
            torch.cuda.synchronize()
            
            start = time.time()
            result_gpu = apply_convolution_gpu(test_image, kernel, device)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            speedup = cpu_time / gpu_time
            print(f"{size:<12} {cpu_time:<12.6f} {gpu_time:<12.6f} {speedup:<10.2f}x")
        else:
            print(f"{size:<12} {cpu_time:<12.6f} {'N/A':<12} {'N/A':<10}")
    
    print("\n💡 观察:")
    print("  - 图像越大,GPU加速效果越明显")
    print("  - 小图像可能GPU反而更慢(数据传输开销)")
    print("  - 对于图像处理,GPU可以提供几十倍加速")


def visualize_all_filters(image, kernels, device):
    """可视化所有滤波效果"""
    print_section("应用所有滤波器")
    
    # 准备画布
    n_kernels = len(kernels)
    cols = 4
    rows = (n_kernels + cols) // cols + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten()
    
    # 显示原图
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像', fontsize=12)
    axes[0].axis('off')
    
    # 应用每个卷积核
    for idx, (name, kernel) in enumerate(kernels.items(), 1):
        print(f"应用 {name}...")
        
        # 使用GPU加速
        if torch.cuda.is_available():
            result = apply_convolution_gpu(image, kernel, device)
        else:
            result = apply_convolution_cpu(image, kernel)
        
        # 特殊处理:边缘检测合并
        if 'Sobel边缘(X)' in name or 'Prewitt边缘(X)' in name:
            # 保存X方向结果
            edge_x = result
            # 获取Y方向
            y_name = name.replace('(X)', '(Y)')
            if y_name in kernels:
                edge_y = apply_convolution_gpu(image, kernels[y_name], device) if torch.cuda.is_available() else apply_convolution_cpu(image, kernels[y_name])
                result = np.sqrt(edge_x**2 + edge_y**2)
                name = name.replace('(X)', '(合并)')
        
        # 归一化
        result = np.clip(result, 0, 1) if '边缘' not in name else np.abs(result)
        
        # 显示
        axes[idx].imshow(result, cmap='gray')
        axes[idx].set_title(name, fontsize=10)
        axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(kernels) + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_processing_results.png', dpi=150, bbox_inches='tight')
    print("\n处理结果已保存为 image_processing_results.png")


def interactive_demo(image, device):
    """交互式演示:调整滤波强度"""
    print_section("交互式演示:可调滤波器")
    
    print("\n演示:可调节的模糊强度")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 不同的模糊强度
    blur_sizes = [3, 5, 7, 9, 15, 21]
    
    for idx, kernel_size in enumerate(blur_sizes):
        # 创建均值滤波核
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        
        # 应用滤波
        if torch.cuda.is_available():
            result = apply_convolution_gpu(image, kernel, device)
        else:
            result = apply_convolution_cpu(image, kernel)
        
        # 显示
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(result, cmap='gray')
        axes[row, col].set_title(f'均值滤波 {kernel_size}x{kernel_size}', fontsize=12)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('blur_intensity_demo.png', dpi=150, bbox_inches='tight')
    print("模糊强度演示已保存为 blur_intensity_demo.png")


def edge_detection_comparison(image, device):
    """边缘检测算法对比"""
    print_section("边缘检测算法对比")
    
    kernels = get_convolution_kernels()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')
    
    # Sobel边缘检测
    sobel_x = apply_convolution_gpu(image, kernels['Sobel边缘(X)'], device) if torch.cuda.is_available() else apply_convolution_cpu(image, kernels['Sobel边缘(X)'])
    sobel_y = apply_convolution_gpu(image, kernels['Sobel边缘(Y)'], device) if torch.cuda.is_available() else apply_convolution_cpu(image, kernels['Sobel边缘(Y)'])
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    axes[0, 1].imshow(np.abs(sobel_x), cmap='gray')
    axes[0, 1].set_title('Sobel X方向', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(sobel_y), cmap='gray')
    axes[0, 2].set_title('Sobel Y方向', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(sobel_magnitude, cmap='gray')
    axes[1, 0].set_title('Sobel梯度幅值', fontsize=12)
    axes[1, 0].axis('off')
    
    # Laplacian边缘检测
    laplacian = apply_convolution_gpu(image, kernels['Laplacian边缘'], device) if torch.cuda.is_available() else apply_convolution_cpu(image, kernels['Laplacian边缘'])
    
    axes[1, 1].imshow(np.abs(laplacian), cmap='gray')
    axes[1, 1].set_title('Laplacian边缘检测', fontsize=12)
    axes[1, 1].axis('off')
    
    # Prewitt边缘检测
    prewitt_x = apply_convolution_gpu(image, kernels['Prewitt边缘(X)'], device) if torch.cuda.is_available() else apply_convolution_cpu(image, kernels['Prewitt边缘(X)'])
    prewitt_y = apply_convolution_gpu(image, kernels['Prewitt边缘(Y)'], device) if torch.cuda.is_available() else apply_convolution_cpu(image, kernels['Prewitt边缘(Y)'])
    prewitt_magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    
    axes[1, 2].imshow(prewitt_magnitude, cmap='gray')
    axes[1, 2].set_title('Prewitt梯度幅值', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('edge_detection_comparison.png', dpi=150, bbox_inches='tight')
    print("边缘检测对比图已保存为 edge_detection_comparison.png")


def batch_processing_demo(device):
    """批量处理演示"""
    print_section("批量图像处理")
    
    batch_sizes = [1, 4, 8, 16]
    image_size = 256
    
    print(f"\n图像尺寸: {image_size}x{image_size}")
    print(f"卷积核: 3x3 高斯模糊\n")
    
    print(f"{'批量大小':<12} {'CPU时间':<12} {'GPU时间':<12} {'加速比':<10}")
    print("-" * 50)
    
    kernel = get_convolution_kernels()['高斯模糊(3x3)']
    
    for batch_size in batch_sizes:
        # 创建批量图像
        images = np.stack([create_test_image(image_size) for _ in range(batch_size)])
        images_tensor = torch.from_numpy(images).unsqueeze(1).float()  # [batch, 1, H, W]
        
        # CPU测试
        start = time.time()
        for i in range(batch_size):
            result = apply_convolution_cpu(images[i], kernel)
        cpu_time = time.time() - start
        
        # GPU测试
        if torch.cuda.is_available():
            images_gpu = images_tensor.cuda()
            kernel_gpu = kernel.cuda()
            
            # 预热
            _ = F.conv2d(images_gpu, kernel_gpu, padding=1)
            torch.cuda.synchronize()
            
            start = time.time()
            results_gpu = F.conv2d(images_gpu, kernel_gpu, padding=1)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            speedup = cpu_time / gpu_time
            print(f"{batch_size:<12} {cpu_time:<12.6f} {gpu_time:<12.6f} {speedup:<10.2f}x")
        else:
            print(f"{batch_size:<12} {cpu_time:<12.6f} {'N/A':<12} {'N/A':<10}")
    
    print("\n💡 观察:")
    print("  - 批量处理比单张循环处理更高效")
    print("  - GPU特别适合批量处理")
    print("  - 批量越大,GPU优势越明显")


def save_test_image(image):
    """保存测试图像"""
    # 转换为PIL图像并保存
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    pil_image.save('test_image.png')
    print("测试图像已保存为 test_image.png")


def main():
    """主函数"""
    print("="*60)
    print("  GPU加速图像滤波器")
    print("  Day 3: Python GPU编程 - PyTorch入门")
    print("="*60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 创建测试图像
    print_section("创建测试图像")
    image = create_test_image(256)
    print(f"图像尺寸: {image.shape}")
    print(f"像素值范围: [{image.min():.2f}, {image.max():.2f}]")
    
    # 保存测试图像
    save_test_image(image)
    
    # 获取卷积核
    kernels = get_convolution_kernels()
    print(f"\n准备了 {len(kernels)} 种卷积核:")
    for name in kernels.keys():
        print(f"  - {name}")
    
    # 性能对比
    compare_performance(image, kernels, device)
    
    # 应用所有滤波器
    visualize_all_filters(image, kernels, device)
    
    # 交互式演示
    interactive_demo(image, device)
    
    # 边缘检测对比
    edge_detection_comparison(image, device)
    
    # 批量处理演示
    batch_processing_demo(device)
    
    # 总结
    print_section("学习总结")
    print("\n✅ 你已经学习了:")
    print("  1. 图像卷积的基本原理")
    print("  2. 多种常用卷积核的效果")
    print("  3. GPU加速图像处理")
    print("  4. CPU与GPU性能对比")
    print("  5. 批量处理的优势")
    
    print("\n🎯 与光刻技术的联系:")
    print("  - 光刻成像仿真本质上是大卷积运算")
    print("  - 掩模图像与光学核的卷积")
    print("  - GPU加速对ILT至关重要")
    print("  - ILT优化需要反复进行光刻仿真")
    
    print("\n📚 下一步:")
    print("  - 运行 performance_benchmark.py 进行详细性能测试")
    print("  - 尝试实现自己的卷积核")
    print("  - 研究快速傅里叶变换(FFT)加速卷积")
    
    print("\n生成的图表:")
    print("  - test_image.png: 测试图像")
    print("  - image_processing_results.png: 所有滤波效果")
    print("  - blur_intensity_demo.png: 模糊强度演示")
    print("  - edge_detection_comparison.png: 边缘检测对比")
    
    print("\n" + "="*60)
    print("练习完成!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
