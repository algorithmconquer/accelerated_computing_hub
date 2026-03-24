"""
GPU环境检查脚本
检查CUDA和GPU的可用性
"""

import torch
import sys

def check_gpu():
    """检查GPU环境"""
    print("=" * 60)
    print("GPU环境检查")
    print("=" * 60)
    
    # 检查PyTorch版本
    print(f"\n【PyTorch信息】")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {sys.version}")
    
    # 检查CUDA是否可用
    print(f"\n【CUDA信息】")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        # 显示每个GPU的详细信息
        for i in range(torch.cuda.device_count()):
            print(f"\n【GPU {i} 信息】")
            props = torch.cuda.get_device_properties(i)
            print(f"  设备名称: {props.name}")
            print(f"  显存总量: {props.total_memory / 1024**3:.2f} GB")
            print(f"  计算能力: {props.major}.{props.minor}")
            print(f"  多处理器数量: {props.multi_processor_count}")
        
        # 测试GPU计算
        print(f"\n【GPU计算测试】")
        try:
            # 矩阵乘法测试
            print("正在执行矩阵乘法测试...")
            size = 5000
            x = torch.randn(size, size, device='cuda')
            y = torch.randn(size, size, device='cuda')
            
            # 预热
            torch.matmul(x, y)
            torch.cuda.synchronize()
            
            # 计时
            import time
            start = time.time()
            for _ in range(10):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            print(f"  矩阵大小: {size}x{size}")
            print(f"  10次矩阵乘法耗时: {elapsed:.4f} 秒")
            print(f"  平均性能: {10 * 2 * size**3 / elapsed / 1e12:.2f} TFLOPS")
            print("  ✓ 矩阵乘法测试成功")
            
            # 显存使用情况
            print(f"\n【显存使用情况】")
            print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
        except RuntimeError as e:
            print(f"  ✗ GPU计算测试失败: {e}")
            return False
        
    else:
        print("\n✗ CUDA不可用!")
        print("\n可能的原因:")
        print("  1. 未安装NVIDIA GPU驱动")
        print("  2. 未安装CUDA Toolkit")
        print("  3. PyTorch版本与CUDA版本不匹配")
        print("\n解决方案:")
        print("  1. 运行 'nvidia-smi' 检查GPU驱动")
        print("  2. 访问 https://developer.nvidia.com/cuda-downloads 安装CUDA")
        print("  3. 重新安装PyTorch: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        return False
    
    # 检查cuFFT(用于光刻仿真)
    print(f"\n【cuFFT检查】")
    try:
        import torch.fft as fft
        x = torch.randn(1024, 1024, device='cuda')
        x_freq = fft.fft2(x)
        x_back = fft.ifft2(x_freq)
        print("  ✓ cuFFT功能正常")
    except Exception as e:
        print(f"  ✗ cuFFT测试失败: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ 环境检查完成!所有测试通过!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
