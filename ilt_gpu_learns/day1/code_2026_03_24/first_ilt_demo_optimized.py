#!/usr/bin/env python3
"""
第一天的ILT演示程序 - 优化版
解决了收敛问题，添加了多项改进
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time


class SimplifiedLithographyModel:
    """简化的光刻成像模型"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.image_size = 256
        
        # 创建点扩散函数(PSF)
        self.psf = self._create_psf()
        
        # 光刻胶参数
        self.resist_threshold = 0.5
        self.resist_contrast = 15.0  # 提高对比度
        
    def _create_psf(self):
        """创建点扩散函数"""
        sigma = 3.0  # 衍射模糊程度
        
        x = np.linspace(-self.image_size/2, self.image_size/2, self.image_size)
        y = np.linspace(-self.image_size/2, self.image_size/2, self.image_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        psf = np.exp(-R**2 / (2 * sigma**2))
        psf = psf / psf.sum()
        
        psf_tensor = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0)
        return psf_tensor.to(self.device)
    
    def forward(self, mask):
        """前向传播"""
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        # 光学成像
        aerial_image = self._convolve_fft(mask, self.psf)
        
        # 光刻胶响应
        wafer_pattern = torch.sigmoid(
            self.resist_contrast * (aerial_image - self.resist_threshold)
        )
        
        return wafer_pattern.squeeze(), aerial_image.squeeze()
    
    def _convolve_fft(self, image, kernel):
        """FFT卷积"""
        height, width = image.shape[2], image.shape[3]
        
        image_fft = torch.fft.rfft2(image)
        kernel_fft = torch.fft.rfft2(kernel, s=(height, width))
        result_fft = image_fft * kernel_fft
        result = torch.fft.irfft2(result_fft, s=(height, width))
        
        return result


class ILTOptimizer:
    """改进的ILT优化器"""
    
    def __init__(self, lithography_model, learning_rate=0.1, device='cuda'):
        self.model = lithography_model
        self.lr = learning_rate
        self.device = device
        
    def optimize(self, target_pattern, num_iterations=500, verbose=True):
        """执行ILT优化 - 改进版"""
        # 确保目标图案维度正确
        if target_pattern.dim() == 4:
            target_pattern = target_pattern.squeeze()
        
        # 改进1：使用参数化方法，通过sigmoid确保mask在[0,1]
        mask_param = torch.nn.Parameter(
            torch.zeros_like(target_pattern, device=self.device)
        )
        
        # 改进2：使用Adam优化器，初始学习率更大
        optimizer = torch.optim.Adam([mask_param], lr=self.lr)
        
        # 改进3：添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=30
        )
        
        history = {
            'loss': [],
            'mask': [],
            'wafer': [],
            'lr': []
        }
        
        print(f"\n开始ILT优化（改进版）...")
        print(f"设备: {self.device}")
        print(f"初始学习率: {self.lr}")
        print(f"迭代次数: {num_iterations}")
        print("-" * 60)
        
        start_time = time.time()
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # 改进4：通过sigmoid参数化mask
            mask = torch.sigmoid(mask_param)
            
            # 前向传播
            wafer_pattern, aerial_image = self.model.forward(mask)
            
            # 改进5：更好的损失函数设计
            # MSE损失
            mse_loss = F.mse_loss(wafer_pattern, target_pattern)
            
            # 改进6：合理的正则化 - 鼓励边界锐利
            # 使用Total Variation正则化，而不是阻止收敛
            tv_loss = 0.001 * self._total_variation_loss(mask)
            
            # 改进7：边缘锐化损失
            edge_loss = 0.01 * self._edge_loss(wafer_pattern, target_pattern)
            
            total_loss = mse_loss + tv_loss + edge_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 改进8：更新学习率调度器
            scheduler.step(total_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            history['loss'].append(total_loss.item())
            history['lr'].append(current_lr)
            
            if verbose and (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"迭代 {i+1:3d}/{num_iterations}: "
                      f"损失 = {total_loss.item():.6f} "
                      f"(MSE={mse_loss.item():.4f}, TV={tv_loss.item():.4f}, Edge={edge_loss.item():.4f}), "
                      f"LR={current_lr:.6f}, "
                      f"时间={elapsed:.2f}s")
        
        # 保存最终结果
        with torch.no_grad():
            mask = torch.sigmoid(mask_param)
            final_wafer, _ = self.model.forward(mask)
            history['mask'] = mask.detach().cpu()
            history['wafer'] = final_wafer.detach().cpu()
        
        total_time = time.time() - start_time
        print("-" * 60)
        print(f"优化完成！总时间: {total_time:.2f}秒")
        print(f"最终损失: {history['loss'][-1]:.6f}")
        print(f"损失降低: {(1 - history['loss'][-1]/history['loss'][0])*100:.2f}%")
        
        return mask.detach(), history
    
    def _total_variation_loss(self, mask):
        """Total Variation正则化 - 平滑但保留边缘"""
        diff_h = torch.abs(mask[1:, :] - mask[:-1, :])
        diff_w = torch.abs(mask[:, 1:] - mask[:, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)
    
    def _edge_loss(self, wafer, target):
        """边缘对齐损失"""
        # Sobel边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        if wafer.dim() == 2:
            wafer = wafer.unsqueeze(0).unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        
        edge_wafer_x = F.conv2d(wafer, sobel_x, padding=1)
        edge_wafer_y = F.conv2d(wafer, sobel_y, padding=1)
        edge_wafer = torch.sqrt(edge_wafer_x**2 + edge_wafer_y**2 + 1e-8)
        
        edge_target_x = F.conv2d(target, sobel_x, padding=1)
        edge_target_y = F.conv2d(target, sobel_y, padding=1)
        edge_target = torch.sqrt(edge_target_x**2 + edge_target_y**2 + 1e-8)
        
        return F.mse_loss(edge_wafer, edge_target)


def create_test_pattern(size=256, pattern_type='contact_hole'):
    """创建测试图案"""
    pattern = np.zeros((size, size))
    
    if pattern_type == 'contact_hole':
        hole_size = 15
        spacing = 40
        start = (size - 4*spacing) // 2
        
        for i in range(5):
            for j in range(5):
                cx = start + j * spacing
                cy = start + i * spacing
                pattern[cy-hole_size//2:cy+hole_size//2, 
                       cx-hole_size//2:cx+hole_size//2] = 1.0
                
    elif pattern_type == 'cross':
        center = size // 2
        width = size // 10
        pattern[center-width//2:center+width//2, center-size//4:center+size//4] = 1.0
        pattern[center-size//4:center+size//4, center-width//2:center+width//2] = 1.0
    
    return torch.from_numpy(pattern).float()


def visualize_results(target, history, save_path='ilt_results_optimized.png'):
    """可视化优化结果"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig)
    
    # 目标图案
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(target.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Target Pattern', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 优化后的掩模
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(history['mask'].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Optimized Mask', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 晶圆图案
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(history['wafer'].squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Wafer Pattern', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 误差图
    ax4 = fig.add_subplot(gs[0, 3])
    error = np.abs(history['wafer'].squeeze().cpu().numpy() - 
                   target.cpu().numpy())
    im = ax4.imshow(error, cmap='hot', vmin=0, vmax=0.3)
    ax4.set_title('Pattern Error', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046)
    
    # 损失曲线
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.plot(history['loss'], linewidth=2, label='Total Loss')
    ax5.set_xlabel('Iteration', fontsize=11)
    ax5.set_ylabel('Loss', fontsize=11)
    ax5.set_title('Optimization Convergence', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_xlim(0, len(history['loss']))
    
    # 学习率变化
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.plot(history['lr'], linewidth=2, color='orange', label='Learning Rate')
    ax6.set_xlabel('Iteration', fontsize=11)
    ax6.set_ylabel('Learning Rate', fontsize=11)
    ax6.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_xlim(0, len(history['lr']))
    ax6.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到: {save_path}")
    
    return fig


def main():
    """主函数"""
    print("="*70)
    print("        逆光刻技术(ILT)演示程序 - 优化版")
    print("="*70)
    
    # 检查GPU
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
    
    # 创建光刻模型
    print("\n[步骤2] 初始化光刻模型...")
    model = SimplifiedLithographyModel(device=device)
    
    # 执行ILT优化 - 使用更大的学习率和更多迭代
    print("\n[步骤3] 开始ILT优化...")
    optimizer = ILTOptimizer(model, learning_rate=0.1, device=device)  # 学习率提高到0.1
    optimized_mask, history = optimizer.optimize(
        target_pattern, 
        num_iterations=500  # 增加迭代次数
    )
    
    # 可视化结果
    print("\n[步骤4] 可视化结果...")
    fig = visualize_results(target_pattern, history, 
                           save_path='ilt_results_optimized.png')
    
    # 打印优化总结
    print("\n" + "="*70)
    print("                        优化总结")
    print("="*70)
    print(f"""
改进措施：
1. ✓ 学习率从0.01提高到0.1（10倍）
2. ✓ 添加学习率调度器（自动降低学习率）
3. ✓ 使用sigmoid参数化（避免梯度消失）
4. ✓ 改进正则化（TV loss替代错误正则化）
5. ✓ 添加边缘损失（更好的边缘对齐）
6. ✓ 增加迭代次数到500次
7. ✓ 提高光刻胶对比度到15.0

预期效果：
- 损失应该降到0.05以下（比之前的0.12降低60%）
- 掩模应该包含清晰的辅助图形
- 晶圆图案应该接近目标图案
""")
    print("="*70)
    
    plt.show()


if __name__ == '__main__':
    main()
