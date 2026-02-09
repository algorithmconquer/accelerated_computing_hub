第 6 章：nvmath-python 简介
nvmath-python（测试版）库将 NVIDIA 数学库的强大功能引入 Python 生态系统。该包旨在提供直观的 Python 式 API，让用户能够在各种执行空间中完全访问 NVIDIA 库提供的所有功能。nvmath-python 与现有的 Python 数组/张量框架无缝协作，并专注于提供这些框架所缺失的功能。

此库旨在满足以下需求：

寻求生产力、与其他库和框架的互操作性以及高性能的研究人员
寻求开箱即用的性能和通过 Python 实现更好可维护性的库/框架开发人员
寻求无需切换到 CUDA 即可获得最高性能的内核开发人员
nvmath-python 特性：

CUDA 数学库的低级绑定
Pythonic 高级 API（主机和设备端）：目前仅限于扩展的矩阵乘法和 FFT
可在 Numba 内核中调用的设备函数
与 NumPy、CuPy 和 PyTorch 张量的互操作性
安装
请使用 nvmath-python 安装指南 在您的硬件和 Python 环境中进行设置。

python
!pip install nvmath-python[cu12,dx,cpu]
在这种安装方法中，我们安装了 CUDA 12 和 CPU 依赖项。此外，我们还安装了主机和设备 API，这仅在 CUDA 12 中受支持。

python
# 使用 CPU 上的 NumPy 数组作为第一版
import numpy as np
import nvmath
import cupy as cp
入门
在 CPU 上原型化您的问题
python
shape = 64, 256, 128
axes = 0, 1
# 创建我们的 ndarray，存储在 CPU 上
a = np.random.rand(*shape) + 1j * np.random.rand(*shape)
# 我们利用主机 API 直接在 CPU 上处理输入
b = nvmath.fft.fft(a, axes=axes)
print(f"Input type = {type(a)}, FFT output type = {type(b)}")
我们可以更改 FFT 的执行方式，这将为我们将数据复制到 GPU，并利用 cuFFT 进行计算。

python
# 复制到 GPU 以使用 cuFFT 处理
b = nvmath.fft.fft(a, axes=axes, execution="cuda")
print(f"Input type = {type(a)}, FFT output type = {type(b)}")
高级模块
nvmath-python 提供常见的开箱即用的高性能操作，无需离开 Python。

包括：

线性代数
快速傅里叶变换
nvmath-python 库支持收尾操作（epilog operations）的融合，提供增强的性能。可用的收尾操作包括：

RELU：应用修正线性单元激活函数。
GELU：应用高斯误差线性单元激活函数。
BIAS：添加偏置向量。
SIGMOID：应用 sigmoid 函数。
TANH：应用双曲正切函数。
这些收尾操作可以组合使用，例如，RELU 和 BIAS 可以融合。自定义收尾操作也可以定义为 Python 函数，并使用 LTO-IR 进行编译。

线性代数
nvmath-python 库提供了一个专门的矩阵乘法接口，用于执行带预定义收尾操作的缩放矩阵-矩阵乘法，作为单个融合内核。这种内核融合可以显著提高效率。

此外，nvmath-python 的有状态 API 将此类操作分解为规划（planning）、自动调优（autotuning）和执行（execution）阶段，从而能够在多次执行中摊销一次性准备成本。

使用 CuPy 数组的 Matmul（无状态）
此示例演示了 CuPy 数组的基本矩阵乘法。

nvmath-python 支持多种框架。每个操作的结果是与传递输入所用框架相同的张量。它也位于与输入相同的设备上。

此示例是无状态的，因为它使用函数式 API。

python
# 准备示例输入数据。
n, m, k = 123, 456, 789
a = cp.random.rand(n, k)
b = cp.random.rand(k, m)
# 执行乘法。
result = nvmath.linalg.advanced.matmul(a, b)
# 同步默认流，因为默认情况下对于 GPU 操作数执行是非阻塞的。
cp.cuda.get_current_stream().synchronize()
# 检查结果是否也是 cupy 数组。
print(f"Inputs were of types {type(a)} and {type(b)} and the result is of type {type(result)}.")
assert isinstance(result, cp.ndarray)
使用 CuPy 数组的 Matmul（有状态）
此示例说明了有状态矩阵乘法对象的使用。有状态 API 是面向对象的。有状态对象可以在多次执行中摊销准备成本。

输入和结果都是 CuPy ndarrays。

python
# 准备示例输入数据。
m, n, k = 123, 456, 789
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)
# 将有状态对象用作上下文管理器，以便自动释放资源。
with nvmath.linalg.advanced.Matmul(a, b) as mm:
    # 规划矩阵乘法。规划返回一系列可以配置的算法，我们将在后面的示例中看到。
    mm.plan()
    # 执行矩阵乘法。
    result = mm.execute()
    # 同步默认流，因为默认情况下对于 GPU 操作数执行是非阻塞的。
    cp.cuda.get_current_stream().synchronize()
    print(f"Input types = {type(a), type(b)}, device = {a.device, b.device}")
    print(f"Result type = {type(result)}, device = {result.device}")
使用 CuPy 数组和 Epilog 的 Matmul（无状态）
此示例演示了收尾操作（epilog）的用法。

Epilog 允许您在矩阵乘法之后在单个融合内核中执行额外的计算。在此示例中，我们将使用 BIAS epilog，它将偏置添加到结果中。

python
m, n, k = 64, 128, 256
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)
bias = cp.random.rand(m, 1)
# 使用 BIAS epilog 执行乘法。
epilog = nvmath.linalg.advanced.MatmulEpilog.BIAS
result = nvmath.linalg.advanced.matmul(a, b, epilog=epilog, epilog_inputs={"bias": bias})
# 同步默认流，因为默认情况下对于 GPU 操作数执行是非阻塞的。
cp.cuda.get_current_stream().synchronize()
print(f"Inputs were of types {type(a)} and {type(b)}, the bias type is {type(bias)}, and the result is of type {type(result)}.")
快速傅里叶变换
在 NVIDIA cuFFT 库的支持下，nvmath-python 提供了一套强大的 API 来执行 N 维离散傅里叶变换。这些包括复数到复数、复数到实数和实数到复数情况的正向和逆向变换。这些操作支持多种精度，既有主机 API 也有设备 API。

用户可以向选定的 nvmath-python 操作（如 FFT）提供用 Python 编写的回调函数，这会导致融合内核并能显著提高性能。高级用户可以受益于 nvmath-python 设备 API，这些 API 可以将 FFT 和矩阵乘法等核心数学操作融合到单个内核中，使性能接近理论最大值。

python
# 准备示例输入数据。
m, n, k = 64, 128, 256
a = cp.random.rand(m, k)
b = cp.random.rand(k, n)
bias = cp.random.rand(m, 1)
# 使用 BIAS epilog 执行乘法。
epilog = nvmath.linalg.advanced.MatmulEpilog.BIAS
result = nvmath.linalg.advanced.matmul(a, b, epilog=epilog, epilog_inputs={"bias": bias})
# 同步默认流，因为默认情况下对于 GPU 操作数执行是非阻塞的。
cp.cuda.get_current_stream().synchronize()
print(f"Inputs were of types {type(a)} and {type(b)}, the bias type is {type(bias)}, and the result is of type {type(result)}.")
使用 CuPy 数组的 FFT
FFT 操作的输入和结果都是 CuPy ndarrays，使得 nvmath-python 和 CuPy 之间的互操作性变得毫不费力。

python
shape = 512, 256, 512
axes = 0, 1
a = cp.random.rand(*shape, dtype=cp.float64) + 1j * cp.random.rand(*shape, dtype=cp.float64)
# 沿指定轴进行正向 FFT，沿补集轴进行批处理。
b = nvmath.fft.fft(a, axes=axes)
# 沿指定轴进行逆向 FFT，沿补集轴进行批处理。
c = nvmath.fft.ifft(b, axes=axes)
# 同步默认流
cp.cuda.get_current_stream().synchronize()
print(f"Input type = {type(a)}, device = {a.device}")
print(f"FFT output type = {type(b)}, device = {b.device}")
print(f"IFFT output type = {type(c)}, device = {c.device}")
带回调的 FFT
用户定义的函数可以编译为 LTO-IR 格式，并作为收尾函数（epilog）或前导函数（prolog）提供给 FFT 操作，从而实现链接时优化（Link-Time Optimization）和融合。

此示例展示了如何通过将 Python 回调函数作为前导函数提供给 IFFT 操作来执行卷积。

python
B, N = 256, 1024
a = cp.random.rand(B, N, dtype=cp.float64) + 1j * cp.random.rand(B, N, dtype=cp.float64)
# 创建用作滤波器的数据。
filter_data = cp.sin(a)
# 为逆 FFT 定义前导函数。
# 卷积对应于频域中的逐点乘法。
def convolve(data_in, offset, filter_data, unused):
    # 请注意，我们使用单个 `offset` 整数访问 `data_out` 和 `filter_data`，
    # 尽管输入和 `filter_data` 是二维张量（样本批次）。
    # 必须确保此处访问的两个数组具有相同的内存布局。
    return data_in[offset] * filter_data[offset] / N
# 将前导函数编译为 LTO-IR。
with cp.cuda.Device():
    prolog = nvmath.fft.compile_prolog(convolve, "complex128", "complex128")
# 执行正向 FFT，然后执行逆向 FFT，并将滤波器作为前导函数应用。
r = nvmath.fft.fft(a, axes=[-1])
r = nvmath.fft.ifft(r, axes=[-1], prolog={
        "ltoir": prolog,
        "data": filter_data.data.ptr
    })
低级模块
提供对 CUDA 内部和 CUDA C 数学库的直接访问。

包括：

设备 API
数学库绑定
还可以访问主机 API（以及带回调的主机 API），但这里我们将重点关注设备端。

设备 API
nvmath-python 的设备模块 nvmath.device 通过 cuFFTDx、cuBLASDx 和 cuRAND 的设备 API 提供与 NVIDIA 高性能计算库的集成。这些库的详细文档可分别在 cuFFTDx、cuBLASDx 和 cuRAND 设备 API 中找到。

用户可以通过以下两种方式利用设备模块：

Numba 扩展：用户可以通过 Numba 访问这些设备 API，利用特定的扩展简化定义函数、查询设备特性和调用设备函数的过程。
第三方 JIT 编译器：这些 API 也可通过其他 JIT 编译器中的低级接口获得，允许高级用户直接使用原始设备代码。
此示例展示了如何使用 cuRAND 从正态分布中采样单精度值。

python
from numba import cuda
from nvmath.device import random
compiled_apis = random.Compile()
threads, blocks = 128, 128
nthreads = blocks * threads
states = random.StatesPhilox4_32_10(nthreads)
# 接下来，定义并启动一个设置内核，它将使用 nvmath.device.random.init 函数初始化状态。
@cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
def setup(states):
    i = cuda.grid(1)
    random.init(1234, i, 0, states[i])
setup[blocks, threads](states)
# 准备好状态数组后，您可以使用诸如 nvmath.device.random.normal2 之类的采样器
# 在内核中采样随机值。
@cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
def kernel(states):
    i = cuda.grid(1)
    random_values = random.normal2(states[i])
数学库绑定
NVIDIA 数学库的 C API 的低级 Python 绑定在 nvmath.bindings 中的相应模块下公开。要访问 Python 绑定，请使用相应库的模块。在底层，nvmath-python 会为您懒加载处理库的运行时链接。

当前支持的库及其相应的模块名称如下：

cuBLAS (nvmath.bindings.cublas)
cuBLASLt (nvmath.bindings.cublasLt)
cuFFT (nvmath.bindings.cufft)
cuRAND (nvmath.bindings.curand)
cuSOLVER (nvmath.bindings.cusolver)
cuSOLVERDn (nvmath.bindings.cusolverDn)
cuSPARSE (nvmath.bindings.cusparse)
将库函数名称从 C 转换为 Python 的指南在此处记录：https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html

参考链接
nvmath-python 主页：https://developer.nvidia.com/nvmath-python

nvmath-python 文档：https://docs.nvidia.com/cuda/nvmath-python/latest/index.html

nvmath-python GitHub 仓库：https://developer.nvidia.com/nvmath-python

使用 nvmath-python 将 Epilog 操作与矩阵乘法融合的博客文章：https://developer.nvidia.com/blog/fusing-epilog-operations-with-matrix-multiplication-using-nvmath-python/

示例
nvmath-python GitHub 仓库中提供了一套完整的示例：https://github.com/NVIDIA/nvmath-python/tree/main/examples

