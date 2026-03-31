"""
线性代数演示程序

本程序演示线性代数在光刻技术中的应用,包括:
1. 矩阵创建与基本运算
2. 特征值与特征向量计算
3. 矩阵分解(SVD)
4. 光刻掩模的矩阵表示

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
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU型号: {gpu_name}")
        print(f"GPU显存: {total_memory:.2f} GB")
    else:
        device = torch.device('cpu')
        print("未检测到GPU,将使用CPU运行")
    
    print()
    return device


def matrix_basics(device):
    """
    演示矩阵基础操作
    
    在光刻中,矩阵用于表示:
    - 掩模图案(二值矩阵)
    - 光强分布(灰度矩阵)
    - 光学传输函数(复数矩阵)
    """
    print("="*60)
    print("1. 矩阵基础操作")
    print("="*60)
    
    # 创建一个简单的掩模图案矩阵
    print("\n创建掩模图案矩阵:")
    mask = torch.tensor([
        [0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0]
    ], dtype=torch.float32, device=device)
    
    print(f"掩模矩阵:\n{mask}")
    print(f"\n掩模形状: {mask.shape}")
    print(f"透光区域占比: {(mask.sum() / mask.numel() * 100):.2f}%")
    
    # 矩阵运算
    print("\n基本矩阵运算:")
    
    # 1. 矩阵加法(增加亮度)
    brightened = mask + 0.2
    print(f"1. 矩阵加法(亮度+0.2): 最小值={brightened.min():.2f}, 最大值={brightened.max():.2f}")
    
    # 2. 矩阵标量乘法(降低亮度)
    dimmed = mask * 0.5
    print(f"2. 矩阵乘法(亮度×0.5): 最小值={dimmed.min():.2f}, 最大值={dimmed.max():.2f}")
    
    # 3. 矩阵转置
    transposed = mask.T
    print(f"3. 矩阵转置: 形状从{mask.shape}变为{transposed.shape}")
    
    # 4. 矩阵求和
    total_sum = mask.sum()
    row_sum = mask.sum(dim=1)  # 按行求和
    col_sum = mask.sum(dim=0)  # 按列求和
    print(f"4. 矩阵求和: 总和={total_sum:.0f}, 行和={row_sum}, 列和={col_sum}")
    
    # 5. 矩阵乘法(需要合适形状)
    # 创建一个简单的变换矩阵
    transform = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    # 提取掩模的一部分进行矩阵乘法
    mask_part = mask[:3, :3]
    result = torch.matmul(mask_part, transform)
    print(f"5. 矩阵乘法: {mask_part.shape} @ {transform.shape} = {result.shape}")
    
    return mask


def create_mask_patterns(device, size=64):
    """
    创建不同类型的掩模图案
    
    在实际光刻中,掩模图案多种多样:
    - 接触孔(矩形)
    - 通孔(圆形)
    - 金属线(线条)
    - 复杂图形(组合)
    """
    print("\n" + "="*60)
    print("2. 创建不同类型的掩模图案")
    print("="*60)
    
    masks = {}
    
    # 1. 矩形孔径(接触孔)
    print("\n1) 创建矩形孔径掩模:")
    rect_mask = torch.zeros(size, size, device=device)
    rect_mask[20:40, 15:45] = 1.0
    masks['rectangle'] = rect_mask
    print(f"   矩形孔径: 位置[20:40, 15:45], 面积={(40-20)*(45-15)}像素")
    
    # 2. 圆形孔径(通孔)
    print("\n2) 创建圆形孔径掩模:")
    circle_mask = torch.zeros(size, size, device=device)
    center = (size//2, size//2)
    radius = 15
    y, x = torch.meshgrid(torch.arange(size, device=device), 
                           torch.arange(size, device=device), indexing='ij')
    circle_mask[(x - center[0])**2 + (y - center[1])**2 <= radius**2] = 1.0
    masks['circle'] = circle_mask
    print(f"   圆形孔径: 中心{center}, 半径{radius}, 面积≈{3.14159*radius**2:.0f}像素")
    
    # 3. 线条图案(金属线)
    print("\n3) 创建线条图案掩模:")
    line_mask = torch.zeros(size, size, device=device)
    # 水平线
    line_mask[15:20, 10:50] = 1.0
    line_mask[30:35, 10:50] = 1.0
    line_mask[45:50, 10:50] = 1.0
    # 垂直线
    line_mask[10:50, 15:20] = 1.0
    line_mask[10:50, 35:40] = 1.0
    masks['lines'] = line_mask
    print(f"   线条图案: 多条水平和垂直线")
    
    # 4. 复杂图案(组合)
    print("\n4) 创建复杂图案掩模:")
    complex_mask = torch.zeros(size, size, device=device)
    # L形
    complex_mask[10:40, 10:15] = 1.0
    complex_mask[35:40, 10:35] = 1.0
    # T形
    complex_mask[15:20, 45:55] = 1.0
    complex_mask[15:35, 48:53] = 1.0
    masks['complex'] = complex_mask
    print(f"   复杂图案: L形和T形组合")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    for idx, (name, mask) in enumerate(masks.items()):
        ax = axes[idx // 2, idx % 2]
        ax.imshow(mask.cpu(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{name} mask')
        ax.axis('off')
        
        # 添加网格
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Different Mask Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mask_patterns.png', dpi=150, bbox_inches='tight')
    print(f"\n掩模图案已保存为 mask_patterns.png")
    
    return masks


def eigenvalue_analysis(device):
    """
    演示特征值与特征向量分析
    
    在光刻中的应用:
    - SOCS方法中分解部分相干光源
    - 分析光学系统的特性
    - 理解图像的主要变化方向
    """
    print("\n" + "="*60)
    print("3. 特征值与特征向量分析")
    print("="*60)
    
    # 创建一个对称矩阵(代表光学传输函数的TCC矩阵)
    print("\n创建对称矩阵(类似光学传输函数TCC):")
    A = torch.tensor([
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 0.5],
        [1.0, 0.5, 2.0]
    ], device=device)
    
    print(f"矩阵 A:\n{A}")
    
    # 计算特征值和特征向量
    print("\n计算特征值和特征向量:")
    eigenvalues, eigenvectors = torch.linalg.eigh(A)  # 对称矩阵用eigh
    
    print(f"特征值(从小到大): {eigenvalues}")
    print(f"特征向量(每列对应一个特征值):\n{eigenvectors}")
    
    # 验证特征值方程: A * v = λ * v
    print("\n验证特征值方程 A * v = λ * v:")
    errors = []
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lambda_val = eigenvalues[i]
        Av = torch.matmul(A, v)
        lambda_v = lambda_val * v
        error = torch.norm(Av - lambda_v).item()
        errors.append(error)
        print(f"  特征值{i+1}: λ={lambda_val:.4f}, 误差={error:.8f}")
    
    # 物理意义解释
    print("\n物理意义解释:")
    print(f"  - 最大特征值 {eigenvalues[-1]:.4f}: 主导模式,对系统影响最大")
    print(f"  - 中间特征值 {eigenvalues[1]:.4f}: 次要模式")
    print(f"  - 最小特征值 {eigenvalues[0]:.4f}: 微弱模式,影响最小")
    print("  在SOCS方法中,只保留大的特征值对应的光源模式")
    
    # 可视化特征向量
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i in range(3):
        v = eigenvectors[:, i].cpu().numpy()
        axes[i].bar(range(3), v)
        axes[i].set_xlabel('Component')
        axes[i].set_ylabel('Value')
        axes[i].set_title(f'Eigenvector {i+1}\nλ={eigenvalues[i]:.4f}')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Eigenvectors Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eigenvectors.png', dpi=150, bbox_inches='tight')
    print(f"\n特征向量可视化已保存为 eigenvectors.png")
    
    return eigenvalues, eigenvectors


def matrix_decomposition(device):
    """
    演示矩阵分解(SVD)
    
    在光刻中的应用:
    - 图像压缩与降维
    - SOCS方法的核心
    - 信号分离
    """
    print("\n" + "="*60)
    print("4. 矩阵分解(SVD)")
    print("="*60)
    
    # 创建一个图像矩阵
    size = 64
    image = torch.zeros(size, size, device=device)
    
    # 添加一些图案
    image[20:40, 15:45] = 1.0  # 矩形
    center = (50, 50)
    radius = 10
    y, x = torch.meshgrid(torch.arange(size, device=device), 
                           torch.arange(size, device=device), indexing='ij')
    image[(x - center[0])**2 + (y - center[1])**2 <= radius**2] = 1.0
    
    print(f"创建图像矩阵: {image.shape}")
    
    # SVD分解
    print("\n执行SVD分解: A = U @ S @ V^T")
    U, S, V = torch.linalg.svd(image)
    
    print(f"U的形状: {U.shape}  (左奇异向量)")
    print(f"S的形状: {S.shape}  (奇异值)")
    print(f"V的形状: {V.shape}  (右奇异向量)")
    
    # 奇异值分析
    print(f"\n前10个奇异值: {S[:10]}")
    total_energy = (S ** 2).sum()
    cumulative_energy = torch.cumsum(S ** 2, dim=0)
    energy_ratio = cumulative_energy / total_energy
    
    print(f"\n能量占比:")
    print(f"  前5个奇异值: {energy_ratio[4]*100:.2f}%")
    print(f"  前10个奇异值: {energy_ratio[9]*100:.2f}%")
    print(f"  前20个奇异值: {energy_ratio[19]*100:.2f}%")
    
    # 使用不同数量的奇异值重构图像
    ranks = [5, 10, 20, 50]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(image.cpu(), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 重构图像
    for idx, rank in enumerate(ranks):
        # 使用前rank个奇异值重构
        U_k = U[:, :rank]
        S_k = torch.diag(S[:rank])
        V_k = V[:rank, :]
        
        reconstructed = U_k @ S_k @ V_k
        
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        axes[row, col].imshow(reconstructed.cpu(), cmap='gray')
        axes[row, col].set_title(f'Rank-{rank} Approximation\n({energy_ratio[rank-1]*100:.1f}% energy)')
        axes[row, col].axis('off')
    
    # 奇异值衰减曲线
    axes[1, 2].semilogy(S.cpu(), 'b-', linewidth=2)
    axes[1, 2].set_xlabel('Index')
    axes[1, 2].set_ylabel('Singular Value (log scale)')
    axes[1, 2].set_title('Singular Value Decay')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('SVD Decomposition and Reconstruction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('svd_decomposition.png', dpi=150, bbox_inches='tight')
    print(f"\nSVD分解可视化已保存为 svd_decomposition.png")
    
    return U, S, V


def matrix_operations_performance(device):
    """
    测试不同矩阵运算的性能
    
    了解哪些操作在GPU上更高效
    """
    print("\n" + "="*60)
    print("5. 矩阵运算性能测试")
    print("="*60)
    
    sizes = [100, 500, 1000, 2000]
    
    print(f"\n测试矩阵大小: {sizes}")
    print("\n测试结果:")
    
    results = []
    for size in sizes:
        # 创建随机矩阵
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        
        # 矩阵乘法
        torch.cuda.synchronize()
        import time
        start = time.time()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        matmul_time = time.time() - start
        
        # 矩阵转置
        start = time.time()
        D = A.T
        torch.cuda.synchronize()
        transpose_time = time.time() - start
        
        # 矩阵求和
        start = time.time()
        total = A.sum()
        torch.cuda.synchronize()
        sum_time = time.time() - start
        
        # 特征值分解(只测试小矩阵)
        if size <= 500:
            start = time.time()
            eigenvalues, _ = torch.linalg.eigh(A @ A.T)  # 对称化
            torch.cuda.synchronize()
            eig_time = time.time() - start
        else:
            eig_time = None
        
        result = {
            'size': size,
            'matmul': matmul_time,
            'transpose': transpose_time,
            'sum': sum_time,
            'eig': eig_time
        }
        results.append(result)
        
        print(f"\n矩阵大小 {size}x{size}:")
        print(f"  矩阵乘法: {matmul_time*1000:.2f} ms")
        print(f"  矩阵转置: {transpose_time*1000:.2f} ms")
        print(f"  矩阵求和: {sum_time*1000:.2f} ms")
        if eig_time:
            print(f"  特征值分解: {eig_time*1000:.2f} ms")
    
    return results


def main():
    """主函数"""
    print("\n" + "="*60)
    print("Day 4: 线性代数演示程序")
    print("="*60)
    
    # 检查GPU
    device = check_gpu()
    
    # 1. 矩阵基础操作
    mask = matrix_basics(device)
    
    # 2. 创建不同类型的掩模图案
    masks = create_mask_patterns(device)
    
    # 3. 特征值分析
    eigenvalues, eigenvectors = eigenvalue_analysis(device)
    
    # 4. 矩阵分解(SVD)
    U, S, V = matrix_decomposition(device)
    
    # 5. 性能测试
    results = matrix_operations_performance(device)
    
    print("\n" + "="*60)
    print("所有演示完成!")
    print("="*60)
    print("\n生成的文件:")
    print("  1. mask_patterns.png - 不同类型的掩模图案")
    print("  2. eigenvectors.png - 特征向量可视化")
    print("  3. svd_decomposition.png - SVD分解结果")
    print("\n请查看这些图像以加深理解!")


if __name__ == "__main__":
    main()
