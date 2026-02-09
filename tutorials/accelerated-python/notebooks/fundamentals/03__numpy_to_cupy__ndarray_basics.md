使用 CuPy 进行加速计算
目录
创建数组：CPU vs. GPU
基本操作
顺序操作与内存
复杂操作（线性代数）
与设备无关的代码（NumPy 分派）
设备管理
练习 - NumPy 到 CuPy
第 1 部分
第 2 部分
让我们转向使用 CuPy 来实现高级数组功能。

什么是 CuPy？
CuPy 是一个实现了我们熟悉的 NumPy API 的库，但它运行在 GPU 上（后端使用 CUDA C++）。

为什么要使用它？

零摩擦： 如果您熟悉 NumPy，您就已经熟悉 CuPy 了。
速度： 它为数组操作提供开箱即用的 GPU 加速。
易用性： 您通常只需将 import numpy as np 更改为 import cupy as cp，即可将 CPU 代码移植到 GPU。
1. 创建数组：CPU vs. GPU
让我们比较一下在 CPU 和 GPU 上创建一个大型 3D 数组（大小约为 2GB）的性能。

我们将在 CPU 上使用 np.ones，在 GPU 上使用 cp.ones。

我们可以看到，在 GPU 上创建这个数组比在 CPU 上快得多！

关于 cupyx.profiler.benchmark：

我们使用 CuPy 内置的 benchmark 工具来为 GPU 操作计时。这很重要，因为 GPU 操作是异步的——当您调用 CuPy 函数时，CPU 会将任务放入 GPU 的"待办事项列表"（流）中，并立即继续执行下一步，而不等待。

benchmark 函数为我们处理了正确进行 GPU 计时的所有复杂性：

它会自动同步 GPU 流以获得准确的测量结果。
它会运行预热迭代以避免冷启动开销。
它分别报告 CPU 和 GPU 的时间。
这使其成为为 CuPy 代码计时的推荐方式，既准确又方便。

2. 基本操作
数学运算的语法是相同的。让我们将数组中的每个值乘以 5。

GPU 以相同的代码更快地完成了此操作。

顺序操作与内存
现在让我们依次执行几个操作，如果在 Numba 示例中没有显式管理内存，这种情况会受到内存传输时间的影响。

GPU 运行得更快，即使我们没有显式管理内存。这是因为 CuPy 正在透明地为我们处理所有这些。

3. 复杂操作（线性代数）
GPU 在线性代数方面表现出色。让我们看看奇异值分解 (SVD)，这是一个计算量很大的 $O(N^3)$ 操作。

GPU 再次以完全相同的 API 优于 CPU！

与设备无关的代码（NumPy 分派）
CuPy 的一个关键特性是，许多 NumPy 函数无需更改代码即可在 CuPy 数组上工作。

当您将 CuPy GPU 数组 (x_gpu) 传递给支持 __array_function__ 协议的 NumPy 函数（例如 np.linalg.svd）时，NumPy 会检测到 CuPy 输入并将操作委托给 CuPy 自己的实现，该实现在 GPU 上运行。

这使您可以使用标准的 np.* 语法编写代码，并使其无缝地在 CPU 或 GPU 上运行——只要 CuPy 为该函数实现了覆盖。

CuPy 还可以保护您免受隐藏的性能损失：它禁止隐式的 GPU → CPU 复制，当 NumPy 尝试在后台将 cupy.ndarray 转换为 numpy.ndarray 时，会引发 TypeError。这确保了所有设备到主机的传输都是显式且有意的，而非静默发生。

4. 设备管理
如果您有多个 GPU，CuPy 使用"当前设备"上下文的概念。

您可以使用 with 语句来确保在特定显卡上创建特定的数组（例如，GPU 0 与 GPU 1）。

注意： CuPy 函数通常要求所有输入数组位于同一设备上。传递存储在非当前设备上的数组可能会根据硬件配置工作，但通常不建议这样做，因为它可能性能不佳。

练习 - NumPy 到 CuPy
第 1 部分
让我们用与上一个笔记本相同的数据管道来测试"即插即用替换"的理念。具体而言，下面的单个代码块执行以下步骤：

生成一个庞大的数据集（5000 万个元素）。
使用一个繁重的操作（排序）来处理它。
操作形状并对数据进行归一化（广播）。
验证结果的完整性。
TODO:

使用 xp = np（CPU 模式）运行下面的单元格。记下基准测试输出。
将设置行更改为 xp = cp（GPU 模式）。再次运行它。
观察完全相同的逻辑如何在 GPU 上使用 CuPy 显著更快地运行，同时保留 NumPy 的实现特性。
注意：我们使用 cupyx.profiler.benchmark 进行计时，它会自动处理 GPU 同步。

TODO: 使用 CuPy 数组时，尝试将 xp.testing.assert_allclose 更改为 np.testing.assert_allclose。会发生什么，为什么？

第 2 部分
我们现在将创建一个代表正弦波的庞大数据集（5000 万个点），看看 GPU 与 CPU 相比排序的速度有多快。

TODO:

生成数据： 创建一个 NumPy 数组 (y_cpu) 和一个 CuPy 数组 (y_gpu)，表示从 $0$ 到 $2\pi$ 的 $\sin(x)$，包含 50,000,000 个点。
基准测试 CPU 和 GPU： 使用 cupyx.profiler 中的 benchmark() 来测量 np.sort 和 cp.sort。
附加题：使用不同的数组大小进行基准测试，找到 CuPy 和 NumPy 花费相同时间的大小。尝试从 cupyx.profiler.benchmark 的返回值中提取计时数据，并自定义输出的显示方式。您甚至可以制作一个图表。