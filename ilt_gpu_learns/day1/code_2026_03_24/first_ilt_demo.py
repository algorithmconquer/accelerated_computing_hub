#!/usr/bin/env python3
"""
第一天的ILT演示程序
简化版的逆光刻技术(ILT)演示，使用PyTorch GPU加速

这个程序展示了：
1. 简化的光刻成像模型
2. 基于梯度下降的掩模优化
3. GPU加速的计算过程
4. 优化过程的可视化

作者：GPU ILT学习教程
日期：2026-03-24
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time


class SimplifiedLithographyModel:
    """简化的光刻成像模型
    
    真实的ILT使用复杂的Hopkins模型，这里使用简化的卷积模型进行演示。
    主要包括：光源模型 + 掩模传输 + 光学系统 + 光刻胶
    """
    
    def __init__(self, wavelength=193e-9, na=1.35, device='cuda'):
        """
        初始化光刻模型
        
        Args:
            wavelength: 光波长(米)，默认193nm ArF准分子激光
            na: 数值孔径，默认1.35(浸没式光刻)
            device: 计算设备('cuda'或'cpu')
        """
        self.wavelength = wavelength
        self.na = na
        self.device = device
        
        # 图像尺寸和分辨率
        self.image_size = 256
        self.pixel_size = 10e-9  # 10nm像素大小
        
        # 创建点扩散函数(PSF) - 简化的Airy disk模型
        self.psf = self._create_psf()
        
        # 光刻胶阈值参数
        self.resist_threshold = 0.5
        self.resist_contrast = 10.0
        
    def _create_psf(self):
        """创建点扩散函数(PSF)
        
        使用简化的Airy disk模型:
        PSF(r) = [2J₁(kr)/kr]²
        
        在实际光刻中，需要考虑部分相干光源(Hopkins模型)
        """
        # 计算衍射极限
        k = 2 * np.pi / self.wavelength
        sigma = 0.61 * self.wavelength / self.na  # Rayleigh分辨率单位
        
        # 创建网格
        x = np.linspace(-self.image_size/2, self.image_size/2, self.image_size)
        y = np.linspace(-self.image_size/2, self.image_size/2, self.image_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2) * self.pixel_size
        
        # Airy disk (简化版本，使用高斯近似)
        psf = np.exp(-R**2 / (2 * sigma**2))
        psf = psf / psf.sum()  # 归一化
        
        # 转换为PyTorch张量
        psf_tensor = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0)
        return psf_tensor.to(self.device)
    
    def forward(self, mask):
        """前向传播：从掩模到晶圆图案
        
        Args:
            mask: 输入掩模(0-1之间的连续值)
            
        Returns:
            wafer_pattern: 晶圆上的最终图案
        """
        # 1. 掩模传输（简化：直接使用mask值）
        transmitted_light = mask
        
        # 2. 光学成像系统（卷积操作）
        # 使用FFT加速卷积
        aerial_image = self._convolve_fft(transmitted_light, self.psf)
        
        # 3. 光刻胶响应（sigmoid函数模拟）
        wafer_pattern = torch.sigmoid(
            self.resist_contrast * (aerial_image - self.resist_threshold)
        )
        
        return wafer_pattern, aerial_image
    
    def _convolve_fft(self, image, kernel):
        """使用FFT进行快速卷积
        
        在光刻仿真中，FFT卷积是主要的计算瓶颈，
        使用GPU可以大幅加速这一过程。
        """
        # 获取尺寸
        batch, channels, height, width = image.shape
        
        # 填充到避免循环卷积
        pad_height = height // 2
        pad_width = width // 2
        
        # FFT卷积
        image_fft = torch.fft.rfft2(image, s=(height + pad_height, width + pad_width))
        kernel_fft = torch.fft.rfft2(kernel, s=(height + pad_height, width + pad_width))
        
        # 频域相乘
        result_fft = image_fft * kernel_fft
        
        # 逆FFT
        result = torch.fft.irfft2(result_fft, s=(height + pad_height, width + pad_width))
        
        # 裁剪到原始尺寸
        result = result[:, :, :height, :width]
        
        return result


class ILTOptimizer:
    """ILT优化器：使用梯度下降优化掩模
    
    核心思想：
    最小化目标图案与实际晶圆图案之间的差异
    min ||Lith(mask) - target||²
    
    使用Adam优化器进行梯度下降
    """
    
    def __init__(self, lithography_model, learning_rate=0.01, device='cuda'):
        """
        初始化ILT优化器
        
        Args:
            lithography_model: 光刻模型
            learning_rate: 学习率
            device: 计算设备
        """
        self.model = lithography_model
        self.lr = learning_rate
        self.device = device
        
    def optimize(self, target_pattern, num_iterations=200, 
                 mask_init=None, verbose=True):
        """
        执行ILT优化
        
        Args:
            target_pattern: 目标晶圆图案
            num_iterations: 优化迭代次数
            mask_init: 初始掩模(可选)
            verbose: 是否打印进度
            
        Returns:
            optimized_mask: 优化后的掩模
            history: 优化历史记录
        """
        # 初始化掩模
        if mask_init is None:
            # 默认使用目标图案作为初始猜测
            mask = target_pattern.clone().detach().requires_grad_(True)
        else:
            mask = mask_init.clone().detach().requires_grad_(True)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam([mask], lr=self.lr)
        
        # 记录历史
        history = {
            'loss': [],
            'mask': [],
            'wafer': []
        }
        
        print(f"\n开始ILT优化...")
        print(f"设备: {self.device}")
        print(f"迭代次数: {num_iterations}")
        print("-" * 50)
        
        start_time = time.time()
        
        for i in range(num_iterations):
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            wafer_pattern, aerial_image = self.model.forward(mask)
            
            # 计算损失（MSE + 正则化）
            mse_loss = F.mse_loss(wafer_pattern, target_pattern)
            
            # 正则化：鼓励掩模值接近0或1（二元化倾向）
            regularization = 0.01 * torch.mean(mask * (1 - mask))
            
            total_loss = mse_loss + regularization
            
            # 反向传播
            total_loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 约束掩模在[0, 1]范围内
            with torch.no_grad():
                mask.clamp_(0, 1)
            
            # 记录历史
            history['loss'].append(total_loss.item())
            
            if verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"迭代 {i+1:3d}/{num_iterations}: "
                      f"损失 = {total_loss.item():.6f}, "
                      f"MSE = {mse_loss.item():.6f}, "
                      f"时间 = {elapsed:.2f}s")
        
        # 保存最终结果
        with torch.no_grad():
            final_wafer, _ = self.model.forward(mask)
            history['mask'] = mask.detach().cpu()
            history['wafer'] = final_wafer.detach().cpu()
        
        total_time = time.time() - start_time
        print("-" * 50)
        print(f"优化完成！总时间: {total_time:.2f}秒")
        
        return mask.detach(), history


def create_test_pattern(size=256, pattern_type='contact_hole'):
    """创建测试图案
    
    Args:
        size: 图案尺寸
        pattern_type: 图案类型
            - 'contact_hole': 接触孔阵列
            - 'line_space': 线条图案
            - 'complex': 复杂图案
            
    Returns:
        pattern: 目标图案张量
    """
    pattern = np.zeros((size, size))
    
    if pattern_type == 'contact_hole':
        # 接触孔阵列：5x5阵列
        hole_size = 20
        spacing = 40
        start = (size - 4*spacing) // 2
        
        for i in range(5):
            for j in range(5):
                cx = start + j * spacing
                cy = start + i * spacing
                pattern[cy-hole_size//2:cy+hole_size//2, 
                       cx-hole_size//2:cx+hole_size//2] = 1.0
                
    elif pattern_type == 'line_space':
        # 线条图案：密集线条
        line_width = 10
        spacing = 20
        
        for i in range(0, size, spacing):
            pattern[:, i:i+line_width] = 1.0
            
    elif pattern_type == 'complex':
        # 复杂图案：包含不同形状
        # 矩形
        pattern[50:100, 50:150] = 1.0
        # L形
        pattern[150:250, 50:70] = 1.0
        pattern[150:170, 50:150] = 1.0
        # 圆形（近似）
        cx, cy, r = 200, 200, 30
        for i in range(size):
            for j in range(size):
                if (i-cy)**2 + (j-cx)**2 < r**2:
                    pattern[i, j] = 1.0
    
    return torch.from_numpy(pattern).float().unsqueeze(0).unsqueeze(0)


def visualize_results(target, history, save_path='ilt_results.png'):
    """可视化优化结果
    
    Args:
        target: 目标图案
        history: 优化历史
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. 目标图案
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(target.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax1.set_title('目标图案 (Target Pattern)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. 优化后的掩模
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(history['mask'].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax2.set_title('优化后的掩模 (Optimized Mask)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. 实际晶圆图案
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(history['wafer'].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax3.set_title('晶圆图案 (Wafer Pattern)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. 误差图
    ax4 = fig.add_subplot(gs[1, 0])
    error = np.abs(history['wafer'].squeeze().cpu().numpy() - 
                   target.squeeze().cpu().numpy())
    im = ax4.imshow(error, cmap='hot', vmin=0, vmax=0.5)
    ax4.set_title('图案误差 (Pattern Error)', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # 5. 损失曲线
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.plot(history['loss'], linewidth=2)
    ax5.set_xlabel('迭代次数 (Iteration)', fontsize=11)
    ax5.set_ylabel('损失值 (Loss)', fontsize=11)
    ax5.set_title('优化收敛曲线 (Optimization Convergence)', 
                  fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, len(history['loss']))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到: {save_path}")
    
    return fig


def compare_cpu_gpu(target_pattern, num_iterations=100):
    """比较CPU和GPU的计算性能
    
    Args:
        target_pattern: 目标图案
        num_iterations: 迭代次数
    """
    print("\n" + "="*60)
    print("CPU vs GPU 性能对比测试")
    print("="*60)
    
    # CPU测试
    print("\n[CPU测试]")
    model_cpu = SimplifiedLithographyModel(device='cpu')
    optimizer_cpu = ILTOptimizer(model_cpu, device='cpu')
    _, _ = optimizer_cpu.optimize(target_pattern.cpu(), 
                                   num_iterations=num_iterations)
    
    # GPU测试
    if torch.cuda.is_available():
        print("\n[GPU测试]")
        model_gpu = SimplifiedLithographyModel(device='cuda')
        optimizer_gpu = ILTOptimizer(model_gpu, device='cuda')
        _, _ = optimizer_gpu.optimize(target_pattern.cuda(), 
                                       num_iterations=num_iterations)
    else:
        print("\n[警告] CUDA不可用，跳过GPU测试")
    
    print("="*60)


def main():
    """主函数：运行完整的ILT演示"""
    print("="*70)
    print("        逆光刻技术(ILT)演示程序 - 第1天")
    print("        GPU加速的光刻掩模优化")
    print("="*70)
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"\n✓ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = 'cpu'
        print("\n✗ 未检测到GPU，使用CPU计算")
    
    # 创建目标图案
    print("\n[步骤1] 创建测试图案...")
    target_pattern = create_test_pattern(size=256, pattern_type='contact_hole')
    target_pattern = target_pattern.to(device)
    print(f"  图案尺寸: {target_pattern.shape}")
    print(f"  图案类型: 接触孔阵列 (Contact Hole Array)")
    
    # 创建光刻模型
    print("\n[步骤2] 初始化光刻模型...")
    print(f"  光波长: 193 nm (ArF准分子激光)")
    print(f"  数值孔径: 1.35 (浸没式光刻)")
    model = SimplifiedLithographyModel(device=device)
    
    # 执行ILT优化
    print("\n[步骤3] 开始ILT优化...")
    optimizer = ILTOptimizer(model, learning_rate=0.01, device=device)
    optimized_mask, history = optimizer.optimize(
        target_pattern, 
        num_iterations=200
    )
    
    # 可视化结果
    print("\n[步骤4] 可视化结果...")
    fig = visualize_results(target_pattern, history, 
                           save_path='ilt_results.png')
    
    # 性能对比
    compare_cpu_gpu(target_pattern, num_iterations=50)
    
    # 打印学习要点
    print("\n" + "="*70)
    print("                        学习要点总结")
    print("="*70)
    print("""
1. ILT的核心思想：
   - 传统OPC是正向思维：掩模 → 晶圆图案
   - ILT是逆向思维：目标图案 → 优化掩模

2. GPU加速的关键：
   - 光刻仿真的核心是卷积操作（光学成像）
   - FFT卷积将O(N²)降低到O(N log N)
   - GPU的并行计算适合大规模FFT运算

3. 梯度下降优化：
   - 定义损失函数：||实际图案 - 目标图案||²
   - 通过反向传播计算掩模梯度
   - 迭代更新掩模直到收敛

4. 实际应用的复杂性：
   - 真实ILT使用Hopkins部分相干模型
   - 需要考虑多个工艺窗口约束
   - 掩模制造规则检查(MRC)限制

下一步学习：
- Day 2: CUDA编程基础 - 向量运算和矩阵乘法
- Day 3: 光刻成像模型的数学原理
- Day 4: FFT卷积的GPU实现
""")
    print("="*70)
    
    plt.show()


if __name__ == '__main__':
    main()
