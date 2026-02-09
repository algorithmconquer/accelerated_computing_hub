第 5 章：使用 Numba 编写 CUDA 内核
Numba 是一个开源的 JIT（即时）编译器，它将 Python 和 NumPy 代码的一个子集转换为快速的机器码。

Numba 通过遵循 CUDA 执行模型，直接将 Python 代码的一个受限子集编译成 CUDA 内核和设备函数，从而支持 CUDA GPU 编程。使用 Numba 编写的内核似乎可以直接访问 NumPy 数组。NumPy 数组在 CPU 和 GPU 之间自动传输。

Numba 基础
Numba 通过直接将 Python 代码的一个受限子集编译成符合 CUDA 执行模型的 CUDA 内核和设备函数，支持 CUDA GPU 编程。使用 Numba 编写的内核可以直接访问 NumPy 数组。NumPy 数组在 CPU 和 GPU 之间自动传输。Numba 的集成编译系统允许在不大幅改变 Python 语言的情况下，利用 CPU 和 GPU 的特性来编写代码。

安装
在设置 Numba 编程环境之前，请先确保满足以下先决条件（如果您已经按照安装 CuPy 的说明进行操作，则可以跳过这些步骤）：

兼容 CUDA 的 GPU。（有关 NVIDIA GPU 列表，请参见 https://developer.nvidia.com/cuda-gpus）
兼容 CUDA 的 NVIDIA 驱动程序。
CUDA 工具包
参见此处的安装说明：https://numba.readthedocs.io/en/stable/user/installing.html

使用 @cuda.jit 创建内核函数
在 Numba 中，@jit 装饰器用于指定由 Numba 即时编译器优化的函数。在 GPU 的语境下，我们使用名为 @cuda.jit 的版本来指定内核函数，使其能够由 GPU 上的多个线程同时并行执行优化。

python
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
@cuda.jit
def foo(input_array, output_array):
    # 代码块写在这里
这看起来与在 CPU 上使用 Numba 非常相似。

启动内核函数
在运行内核函数之前，需要指定线程块的数量和每个线程块的线程数。这将定义执行网格（Grid）的形状。

python
@cuda.jit
def foo(input_array, output_array):
    # 一维线程块中的线程 ID
    thread_id = cuda.threadIdx.x
    # 一维网格中的线程块 ID
    block_id = cuda.blockIdx.x
    # 线程块宽度，即每个线程块的线程数
    block_width = cuda.blockDim.x
    # 计算数组内部的扁平索引
    i = thread_id + block_id * block_width
    if i < an_array.size:  # 检查数组边界
        output_array[i] = input_array[i]
要调用 foo() 函数，我们必须指定线程块和网格的大小。

python
input = np.asarray(range(10))
output = np.zeros(len(input))
block_threads = 32
grid_blocks = (input.size + (block_threads - 1)) // block_threads
foo[grid_blocks, block_threads](<input, output>)
对于简单的例子，cuda.grid() 函数是管理线程、线程块和网格的一种便捷方式。完整的脚本可以这样重写：

python
import numpy as np
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
input = np.asarray(range(10))
output = np.zeros(len(input))
@cuda.jit
def foo(input_array, output_array):
    i = cuda.grid(1)
    output_array[i] = input_array[i]
    
foo[1, len(input)](<input, output>)
output
注意：当 CUDA 内核执行时，调用会立即返回，而内核执行尚未完成。因此，需要对内核执行进行同步，以确保结果已传输回 CPU。如果不完成此步骤，可能会遇到内存错误，即后续调用试图读取或写入受限内存。使用 cuda.synchronize() 来确保数据一致性。

指定线程和线程块的数量
现在不用太担心这个。只需记住，我们需要指定内核被调用的次数，这由两个数字相乘得出，从而给出总体网格大小。这种设置将确保网格大小拥有足够的线程来处理数据，即使数据大小不是每个线程块线程数的整数倍。

关于每个线程块线程数的经验法则：

最佳线程块大小通常是 32（线程束大小）的倍数。
需要通过分析（Profiling）和基准测试（Benchmarking）来确定最佳值。
入门参考：

NSight 的占用率计算器 (Occupancy Calculator)：https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator
多个来源建议从 128 到 256 之间的数字开始进行调优。
线程块和网格的维度将影响 CUDA 的性能。较大的线程块可以提高共享内存的利用率，并减少启动许多小线程块的开销。然而，过大的线程块可能会减少可以并发执行的线程块数量，从而导致 GPU 利用率不足。为了发挥 GPU 的优势，平衡这一点是必要的。

Numba 与 CuPy 配合使用
CuPy 的 cupy.ndarray 实现了 __cuda_array_interface__，这是与 Numba v0.39.0 或更高版本兼容的 CUDA 数组交换接口（详情参见 Numba 的 CUDA 数组接口文档：https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html）。这意味着您可以将 CuPy 数组传递给使用 Numba 进行 JIT 编译的内核。

在这个例子中，我们使用 cupy 数组代替 numpy 数组：

python
import cupy
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
@cuda.jit
def add(x_array, y_array, output_array):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(start, x_array.shape[0], stride):
                output_array[i] = x_array[i] + y_array[i]
a = cupy.arange(10)
b = a * 2
out = cupy.zeros_like(a)
add[1, 32](<a, b, out>)
print(out)  # => [ 0  3  6  9 12 15 18 21 24 27]
实用参考链接
Numba CUDA GPU：https://numba.readthedocs.io/en/stable/cuda/index.html

CuPy 互操作性指南（包含 Numba）：https://docs.cupy.dev/en/stable/user_guide/interoperability.html

Numba GitHub 仓库：https://github.com/numba/numba

示例：
定义并启动内核函数
python
import numpy as np
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
input = np.asarray(range(10))
output = np.zeros(len(input))
@cuda.jit
def foo(input_array, output_array):
    # 一维线程块中的线程 ID
    thread_id = cuda.threadIdx.x
    # 一维网格中的线程块 ID
    block_id = cuda.blockIdx.x
    # 线程块宽度，即每个线程块的线程数
    block_width = cuda.blockDim.x
    # 计算数组内部的扁平索引
    i = thread_id + block_id * block_width
    if i < input_array.size:  # 检查数组边界
        output_array[i] = input_array[i]
block_threads = 32
grid_blocks = (input.size + (block_threads - 1)) // block_threads
foo[grid_blocks, block_threads](<input, output>)
output
使用 grid() 简化内核函数
python
import numpy as np
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
input = np.asarray(range(10))
output = np.zeros(len(input))
@cuda.jit
def foo(input_array, output_array):
    i = cuda.grid(1)
    if i < input_array.size:
        output_array[i] = input_array[i]
foo[1, len(input)](<input, output>)
output
Numba 与 CuPy 配合使用
python
import cupy
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
@cuda.jit
def add(x_array, y_array, output_array):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(start, x_array.shape[0], stride):
                output_array[i] = x_array[i] + y_array[i]
a = cupy.arange(10)
b = a * 2
out = cupy.zeros_like(a)
add[1, 32](<a, b, out>)
print(out)  # => [ 0  3  6  9 12 15 18 21 24 27]