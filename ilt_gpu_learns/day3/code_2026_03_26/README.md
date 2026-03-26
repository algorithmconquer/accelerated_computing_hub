# Day 3 项目:GPU加速图像滤波器

## 项目概述

本项目通过PyTorch实现GPU加速的图像处理,包括:
- 张量基础操作练习
- 自动微分示例
- GPU加速图像卷积滤波
- CPU与GPU性能对比测试

## 项目结构

```
code_2026_03_26/
├── README.md                    # 本文件
├── tensor_operations.py         # 张量操作练习
├── autograd_demo.py             # 自动微分示例
├── image_filter_gpu.py          # 主程序:GPU图像滤波器
├── performance_benchmark.py     # 性能基准测试
└── expected_outputs.txt         # 预期输出示例
```

## 环境要求

### 必需依赖
- Python 3.8+
- PyTorch 2.0+ (支持CUDA)
- NumPy
- Matplotlib
- Pillow (PIL)

### 检查环境
```bash
# 检查PyTorch和CUDA是否安装正确
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

# 如果CUDA不可用,请安装GPU版本的PyTorch
# 访问 https://pytorch.org/get-started/locally/ 获取安装命令
```

### 安装依赖
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pillow
```

## 运行步骤

### 步骤1:张量操作练习(30分钟)
```bash
cd /Users/zhuwei50/Desktop/workbuddy_claw/gpu_ilt_programming/code_2026_03_26
python tensor_operations.py
```

**预期输出**:
- 各种张量创建方法
- 张量形状操作
- CPU与GPU设备转移
- 基本运算示例

### 步骤2:自动微分示例(45分钟)
```bash
python autograd_demo.py
```

**预期输出**:
- 简单函数梯度计算
- 优化问题求解过程
- 优化路径可视化图
- 多变量优化示例

### 步骤3:GPU图像滤波器(60分钟)
```bash
python image_filter_gpu.py
```

**预期输出**:
- 测试图像创建
- 多种滤波效果(模糊、锐化、边缘检测)
- 处理结果图像 `image_processing_results.png`
- CPU与GPU性能对比

### 步骤4:性能基准测试(45分钟)
```bash
python performance_benchmark.py
```

**预期输出**:
- 不同规模矩阵乘法性能测试
- 不同规模图像卷积性能测试
- 性能对比图表 `performance_comparison.png`
- 详细性能报告

## 学习路径

### 初学者路径(推荐)
1. 先运行 `tensor_operations.py`,理解PyTorch基础
2. 再运行 `autograd_demo.py`,理解自动微分
3. 然后运行 `image_filter_gpu.py`,看到实际应用
4. 最后运行 `performance_benchmark.py`,量化GPU加速效果

### 进阶用户路径
1. 直接运行 `image_filter_gpu.py`,快速看到效果
2. 阅读代码理解实现细节
3. 尝试修改卷积核参数
4. 进行性能调优实验

## 核心代码解析

### 1. 张量创建与设备转移
```python
import torch

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建张量
x = torch.randn(1000, 1000)

# 移动到GPU
x_gpu = x.to(device)

# 或直接在GPU上创建
y_gpu = torch.randn(1000, 1000, device=device)
```

### 2. 自动微分
```python
# 创建需要梯度的变量
x = torch.tensor([2.0], requires_grad=True)

# 构建计算图
y = x ** 2 + 3 * x + 1

# 反向传播
y.backward()

# 查看梯度: dy/dx = 2x + 3
print(x.grad)  # 应该是 2*2 + 3 = 7
```

### 3. GPU加速卷积
```python
import torch.nn.functional as F

# 图像张量: [batch, channel, height, width]
image = torch.randn(1, 1, 256, 256, device='cuda')

# 卷积核: [out_channels, in_channels, kernel_h, kernel_w]
kernel = torch.ones(1, 1, 3, 3, device='cuda') / 9.0  # 均值滤波

# 应用卷积
result = F.conv2d(image, kernel, padding=1)
```

## 常见问题

### Q1: CUDA out of memory错误
**原因**: GPU显存不足

**解决方案**:
```python
# 方法1:减小批量大小
batch_size = 16  # 从32减小到16

# 方法2:清空缓存
torch.cuda.empty_cache()

# 方法3:使用混合精度训练
with torch.cuda.amp.autocast():
    output = model(input)
```

### Q2: 张量在不同设备上无法运算
**错误示例**:
```python
x_cpu = torch.randn(3, 3)
y_gpu = torch.randn(3, 3, device='cuda')
# result = x_cpu + y_gpu  # 报错!
```

**解决方案**:
```python
# 统一到同一设备
y_cpu = y_gpu.cpu()
result = x_cpu + y_cpu

# 或都移到GPU
x_gpu = x_cpu.cuda()
result = x_gpu + y_gpu
```

### Q3: 梯度未清零导致累加
**错误示例**:
```python
for i in range(10):
    loss = model(input)
    loss.backward()
    # 忘记清零梯度!
    optimizer.step()
```

**解决方案**:
```python
for i in range(10):
    optimizer.zero_grad()  # 清零梯度
    loss = model(input)
    loss.backward()
    optimizer.step()
```

### Q4: GPU性能未达预期
**原因**: 
- 数据规模太小,GPU优势不明显
- 未预热GPU
- 内存传输开销大

**解决方案**:
```python
# 1. 预热GPU
_ = model(warmup_data)
torch.cuda.synchronize()

# 2. 使用较大数据规模
size = 5000  # 而不是500

# 3. 减少CPU-GPU数据传输
# 尽量在GPU上完成所有计算,最后一次性传回CPU
```

## 扩展练习

### 练习1:实现自定义卷积核
尝试实现以下效果:
- 高斯模糊
- Laplacian锐化
- 浮雕效果
- 运动模糊

### 练习2:多图像批量处理
```python
# 实现批量图像处理
batch_images = torch.randn(16, 1, 256, 256, device='cuda')

# 应用同一卷积核
results = F.conv2d(batch_images, kernel, padding=1)

# 对比批量处理 vs 循环处理的性能差异
```

### 练习3:实时视频滤波(进阶)
使用OpenCV读取摄像头,实时应用GPU滤波:
```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # 转换为PyTorch张量
    # 应用GPU滤波
    # 显示结果
```

## 性能优化提示

### 1. 使用torch.no_grad()减少内存
```python
# 推理时不需要梯度,可以节省内存
with torch.no_grad():
    output = model(input)
```

### 2. 使用半精度(float16)加速
```python
# 混合精度训练
with torch.cuda.amp.autocast():
    output = model(input)
```

### 3. 固定内存加速数据传输
```python
# 使用pin_memory加速CPU到GPU传输
dataloader = DataLoader(dataset, pin_memory=True)
```

### 4. 使用torch.compile()加速( PyTorch 2.0+)
```python
# 编译模型以加速
model = torch.compile(model)
output = model(input)
```

## 调试技巧

### 1. 检查张量设备
```python
print(tensor.device)  # 查看张量所在设备
```

### 2. 监控GPU使用
```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

### 3. 打印张量形状
```python
print(tensor.shape)  # 查看形状
print(tensor.dtype)  # 查看数据类型
print(tensor.requires_grad)  # 是否需要梯度
```

### 4. 使用断点调试
```python
import pdb; pdb.set_trace()  # 设置断点
```

## 学习目标检查

完成本项目后,你应该能够:

- [ ] 理解PyTorch张量的概念和操作
- [ ] 掌握CPU与GPU之间的数据传输
- [ ] 理解自动微分的工作原理
- [ ] 实现GPU加速的图像卷积
- [ ] 进行CPU与GPU性能对比
- [ ] 理解GPU加速的优势和局限

## 参考资源

- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [PyTorch文档](https://pytorch.org/docs/stable/index.html)
- [CUDA编程指南](https://docs.nvidia.com/cuda/)
- [深度学习与PyTorch入门实战](https://github.com/ShusenTang/Dive-into-DL-PyTorch)

## 作者

GPU ILT学习体系 - Day 3

生成日期: 2026-03-26

## 许可证

本项目代码仅用于学习目的。

---

**开始学习**: 建议按照步骤1-4的顺序运行代码,遇到问题参考常见问题部分。

**提示**: 所有代码都有详细注释,建议先阅读代码理解原理,再运行查看结果。
