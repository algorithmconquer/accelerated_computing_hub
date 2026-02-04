第 4 章：使用 CuPy 进行科学计算
CuPy 是一个与 NumPy 和 SciPy 兼容的数组库，用于使用 Python 进行 GPU 加速计算。CuPy 可以作为无缝替换（drop-in replacement），在 NVIDIA CUDA 或 AMD ROCm 平台上运行现有的 NumPy 和 SciPy 代码。

CuPy 是 Chainer 项目的一部分，但拥有来自包括 NVIDIA 在内的多个组织的维护者。CuPy 实现了人们熟知的 NumPy API，但后端是用 CUDA C++ 编写的。这使得熟悉 NumPy 的开发人员只需切换导入语句，即可快速获得开箱即用的 GPU 加速。

CuPy 基础
CuPy 是一个与 NumPy/SciPy 兼容的数组库，用于使用 Python 进行 GPU 加速计算。CuPy 作为无缝替换，可在 NVIDIA CUDA 或 AMD ROCm 平台上运行现有的 NumPy/SciPy 代码。

CuPy 为 GPU 设备提供了多维数组、稀疏矩阵及相关的例程，所有这些都拥有与 NumPy 和 SciPy 相同的 API。

CuPy 项目的目标是为 Python 用户提供 GPU 加速能力，而无需深入了解底层的 GPU 技术。CuPy 团队致力于提供：

完整的 NumPy 和 SciPy API 覆盖，以实现全面的无缝替换，以及用于最大化性能的高级 CUDA 功能。
成熟且高质量的库，作为所有需要加速的项目（从实验室环境到大规模集群）的基础包。
N 维数组 / cupy.ndarray 数据结构
cupy.ndarray 是 NumPy numpy.ndarray 的 CuPy 对应版本。它为驻留在 CUDA 设备上的固定大小多维数组提供了一个直观的接口。

该类实现了 numpy.ndarray 方法的一个子集。不同之处在于，该类在当前的 GPU 设备上分配数组内容。

内存管理
CuPy 默认使用内存池进行内存分配。内存池通过减少显存分配和 CPU/GPU 同步的开销，显著提高了性能。

CuPy 中有两种不同的内存池：

设备（GPU）内存池：用于 GPU 显存分配。
固定（CPU）内存池（Pinned memory pool）：非交换内存（non-swappable memory），用于 CPU 到 GPU 的数据传输。
在大多数情况下，CuPy 用户不需要关注内存分配和释放的具体细节，但了解 CuPy 内部的这种优化对于基准测试应用程序的性能非常重要。由于内存池中的缓存机制，您可能不会看到显存被完全释放。

CuPy 既提供了控制此类内存的高级 API，也提供了访问 CUDA 内存管理函数的低级 API。

当前设备
CuPy 有一个“当前设备”的概念，即进行数组分配、操作、计算等任务的默认 GPU 设备（默认 id=0）。除多 GPU 功能和设备间复制外，所有 CuPy 操作都在当前活动设备上执行。

通常，CuPy 函数期望数组位于与当前设备相同的设备上。传递存储在非当前设备上的数组可能根据硬件配置能够运行，但由于性能可能不佳，通常不建议这样做。

API
cupy.ndarray
cupy.ndarray 是 CuPy 生态系统的核心，提供了与 numpy.ndarray 直观对应的功能。与 numpy.ndarray 一样，cupy.ndarray 是一个由相同类型和大小的元素组成的固定大小多维容器。

cupy.ufuncs
在 NumPy 中，通用函数（简称 ufunc）被定义为以逐元素方式操作 ndarray 的函数，支持数组广播、类型转换和其他几个标准功能。换句话说，ufunc 是一个“向量化”的包装器，它接收固定数量的特定输入并产生固定数量的特定输出。这些函数主要在 NumPy 数组上运行，是加速 Python 代码最强大的方式之一。（参见 NumPy 通用函数：https://numpy.org/doc/stable/reference/ufuncs.html）

类似地，CuPy 实现了相似的 ufunc，也支持广播、类型转换和输出类型确定。用户可以定义 cupy.ufuncs，使其在 cupy.ndarray 对象上的行为效仿 NumPy ufuncs。

NumPy 和 SciPy 覆盖范围
可用的 NumPy 例程：https://docs.cupy.dev/en/stable/reference/routines.html 可用的 SciPy 例程：https://docs.cupy.dev/en/stable/reference/scipy.html

虽然 CuPy 旨在效仿 NumPy，但在使用 CuPy 时存在一些局限性：

并不是所有的 NumPy 和 SciPy 函数都与 CuPy 兼容。
CuPy 并不总是能提供显著的性能提升。
性能高度依赖于所执行的操作和所使用的硬件。
CuPy 和 NumPy 之间还存在一些差异，可能需要调整代码：

从浮点数到整数的转换行为可能取决于硬件。这是由于 C++ 内部类型转换的局限性造成的。
随机函数差异。NumPy 的 random() 函数不支持 dtype 参数，但 CuPy 底层的随机数生成器 cuRAND 支持。
使用整数数组索引时，CuPy 默认处理越界索引的方式与 NumPy 不同。NumPy 通过抛出错误来处理，而 CuPy 则会进行回绕（wrap around）。
矩阵类型 (numpy.matrix) —— 当从稀疏矩阵计算密集矩阵时（例如 coo_matrix + ndarray），SciPy 倾向于返回 numpy.matrix（numpy.ndarray 的子类）。然而，CuPy 在此类操作中返回 cupy.ndarray。
CuPy 数组不能是非数值型的，例如字符串或对象。
CuPy 中的通用函数（Ufuncs）仅适用于 CuPy 数组或标量。它们不接受其他对象（例如列表或 numpy.ndarray）。
与 NumPy 一样，CuPy 的 RandomState 对象接受数字或完整的 NumPy 数组作为种子。
NumPy 的约简函数（例如 numpy.sum()）返回标量值（例如 numpy.float32）。然而，CuPy 的对应函数返回零维的 cupy.ndarray。
还有更多差异，但以上是最常遇到的。

编码指南
安装
在设置 CuPy 编程环境之前，请确保满足以下先决条件：

兼容 CUDA 的 GPU。（有关 NVIDIA GPU 列表，请参见 https://developer.nvidia.com/cuda-gpus）
兼容 CUDA 的 NVIDIA 驱动程序。
CUDA 工具包
CUDA 工具包的版本将决定您需要安装的 NVIDIA 驱动程序版本。CUDA 工具包兼容包括 Windows、Linux 和 macOS 在内的多种操作系统，但根据您打算使用的 CUDA 工具包版本，可能需要更新操作系统版本。

参见当前的安装说明：https://docs.cupy.dev/en/stable/install.html

最佳实践
在将程序转换为 CuPy 之前，请务必先使用 NumPy 和 SciPy 优化其在 CPU 上的实现。对初始实现进行基准测试将有助于您确定在迁移到 GPU 后是否加速了程序。

要将处理过程从 NumPy 迁移到 CuPy，您需要：

导入 CuPy。
将所有 NumPy 的调用改为 CuPy。
CuPy 覆盖了大部分 NumPy API，因此请先尝试直接转换。
将 NumPy ndarrays 转换为 CuPy ndarrays。
使用 cupy.array() 或 cupy.asarray()。
在 GPU 处理之后，将 CuPy ndarrays 转换回 NumPy ndarrays。
使用 cupy.asnumpy() 或 cupy.ndarray.get()。
例如，这个 NumPy 调用：

python
import numpy as np
x_cpu = np.ones((1000,500,500))
对应于这个 CuPy 调用：

python
import cupy as cp
x_gpu = cp.ones((1000,500,500))
x_cpu = cp.asnumpy(x_gpu)
如果您正在对代码进行基准测试，为了计时的公平性，您需要显式调用 cp.cuda.Stream.null.synchronize()。默认情况下，CuPy 会并发执行 GPU 代码，函数可能在 GPU 运行完之前就已返回。调用 synchronize() 会等待 GPU 完成后再继续。

超越 NumPy 和 SciPy
遗憾的是，NumPy 和 SciPy 并不一定提供您开发软件所需的所有功能。在这种情况下，您需要了解 CuPy 中的一些重要模式：

CuPy 内核编译
CuPy 提供了三个内核编译类。这些类的实例定义了一个 CUDA 内核，可以通过该实例的 __call__ 方法调用：

ElementwiseKernel：像 for 循环一样对数组的每个元素执行。
ReductionKernel：执行映射、约简和后约简函数。
RawKernel：使用原始 CUDA 源代码定义内核，可控制网格大小、线程块大小等。
这些类型中的每一种也都可以使用对应的 @cupyx.jit.* 装饰器定义：@cupyx.jit.elementwisekernel、@cupyx.jit.reductionkernel 和 @cupy.jit.rawkernel。

CuPy 类型泛型内核
如果内核函数中的类型信息是用一个字符定义的，它被视为类型占位符。在整个函数中重复出现的相同字符将被推断为相同类型。这允许创建可重用的泛型内核。

在 GPU 设备间移动
如果您需要在 GPU 之间（从一个设备到另一个设备）移动数据，请使用 with 语句创建上下文。出于能耗或性能考虑，您可能希望在系统中的集成显卡和独立显卡之间切换。

python
import cupy as cp
device_id = 1
# 为设备 1 创建上下文
with cp.cuda.Device(device_id):
   array_on_device1 = cp.array([1, 2, 3, 4, 5])
# 超出上下文范围，在设备 0 上执行
array_on_device0 = cp.array([1, 2, 3, 4, 5])
性能注意事项
将数据从 CPU 移动到 GPU
为了利用 GPU，我们需要通过主板上的 PCI 总线将数据移动到 GPU。这意味着我们需要将数据和代码移动到设备上以执行代码。这样一来，连接 CPU 和 GPU 的 PCI 总线可能成为瓶颈。

在 CPU 和 GPU 之间移动数据会产生一次性的性能开销。

分支
逻辑分支较多的程序需要 CPU。在 CPU 和 GPU 之间切换会产生可能影响性能的成本。对于包含大量 if-then 语句的程序，根据切换处理器的开销，可能更适合使用 CPU。

确保您的函数已向量化，以尽量减少分支。

编译内核函数
当需要调用内核时，CuPy 会编译针对给定参数的维度和数据类型（dtype）优化的内核代码，并将其发送到 GPU 设备执行。进程内会缓存发送到 GPU 设备的内核代码，从而减少后续调用的内核编译时间。

编译内核函数会产生一次性的性能开销。

从当前设备移动数据
通常，CuPy 函数期望数组位于与当前设备相同的设备上。与在 CPU 和 GPU 之间传递数据类似，传递存储在非当前设备上的数组可能会根据硬件配置对性能产生负面影响。

数据从一个设备移动到另一个设备时存在性能权衡。

实用参考链接
更多信息请参见 CuPy 用户指南：https://docs.cupy.dev/en/stable/user_guide/index.html

CuPy API 参考：https://docs.cupy.dev/en/stable/reference/index.html

CuPy GitHub 仓库（包含更多示例）：https://github.com/cupy/cupy

NumPy 用户指南：https://numpy.org/doc/stable/user/

NumPy API 指南：https://numpy.org/doc/stable/reference/index.html

示例
从 NumPy 到 CuPy 的简单转换
python
import numpy as np
x_cpu = np.ones((1000,500,500))
x_cpu
python
import cupy as cp
x_gpu = cp.ones((1000,500,500))
x_cpu = cp.asnumpy(x_gpu)
x_cpu
稍微复杂的 NumPy 到 CuPy 转换
python
import numpy as np
x_cpu = np.random.random((1000, 1000))
x_cpu *= 2 
u, s, v = np.linalg.svd(x_cpu)
# 返回 u, s, v
python
import cupy as cp
x_gpu = cp.random.random((1000, 1000))
x_gpu *= 2 
u, s, v = cp.linalg.svd(x_gpu)
# 返回 u, s, v
添加用户定义的内核函数
python
import cupy
from cupyx import jit
@jit.rawkernel()
def elementwise_copy(x, y, size):
    # 计算线程 ID
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    # 计算总线程数（网格跨度）
    ntid = jit.gridDim.x * jit.blockDim.x
    # 跨步循环处理
    for i in range(tid, size, ntid):
        y[i] = x[i]
size = cupy.uint32(2 ** 22)
x = cupy.random.normal(size=(size,), dtype=cupy.float32)
y = cupy.empty((size,), dtype=cupy.float32)
# 调用方式 1: 显式指定网格和块大小
elementwise_copy((128,), (1024,), (x, y, size))
# 调用方式 2: 使用类似于 CUDA C++ 的 [grid, block] 语法
elementwise_copy[128, 1024](<x, y, size>)
# 验证结果
assert (x == y).all()
