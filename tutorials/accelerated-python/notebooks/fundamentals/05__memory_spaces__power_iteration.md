内存空间与幂迭代
目录
内存空间简介
CPU 基线 (NumPy)
GPU 移植 (CuPy)
优化数据生成
验证与基准测试
附加题
1. 内存空间简介
在我们在 GPU 上实现算法之前，我们必须了解硬件架构。一个异构系统（如您正在使用的系统）由两个不同的内存空间组成：

主机内存 (CPU)： 系统 RAM。由 CPU 访问。
设备内存 (GPU)： 连接到 GPU 的高带宽内存 (HBM)。由 GPU 访问。
CPU 无法直接计算存储在 GPU 上的数据，GPU 也无法直接计算存储在系统 RAM 中的数据。要在 GPU 上执行工作，您必须显式管理数据移动。

主机 $\to$ 设备： 将数据移动到 GPU 进行计算。
语法：x_device = cp.asarray(x_host)
设备 $\to$ 主机： 将结果移回 CPU 以保存到磁盘、使用 Matplotlib 绑图或打印。
语法：y_host = cp.asnumpy(y_device)
隐式传输与同步
理解 CuPy 何时与 CPU 隐式交互至关重要。这些交互会降低性能，因为它们会强制 GPU 暂停（同步）以等待数据移动。

CuPy 在您执行以下操作时会静默传输和同步：

打印 GPU 数组 (
print(gpu_array)
)。
转换 为 Python 标量 (float(gpu_array) 或 .item())。
在布尔上下文中求值 GPU 标量 (if gpu_scalar > 0:)。
任务
为了理解这些概念的含义，让我们使用幂迭代 (Power Iteration) 算法来估计矩阵的主特征值进行实验。

在深入代码之前，让我们先了解我们正在实现的算法背后的数学原理。

幂迭代是一种经典的迭代方法，用于查找方阵 $A$ 的主特征值（绝对值最大的特征值）及其对应的特征向量。

工作原理
核心思想很简单：如果您反复将一个向量乘以矩阵 $A$，该向量最终将收敛到 $A$ 的主特征向量，无论您从哪个初始向量开始（前提是初始向量在主特征向量方向上有一些分量）。

数学步骤
给定一个方阵 $A$ 和一个随机初始向量 $x_0$，算法在每一步 $k$ 按如下方式进行：

1. 矩阵-向量乘法：

我们计算向量的下一个近似值：

$$y = A x_k$$

2. 特征值估计（瑞利商）：

我们使用当前向量估计特征值 $\lambda$。这本质上是将 $y$ 投影到 $x$ 上：

$$\lambda_k = \frac{x_k^T y}{x_k^T x_k} = \frac{x_k^T A x_k}{x_k^T x_k}$$

3. 残差计算（误差检查）：

我们通过计算"残差"（误差）来检查我们与特征向量真正定义（$Ax = \lambda x$）的接近程度：

$$r = ||y - \lambda_k x_k||$$

如果 $r$ 接近 0，则已收敛。

4. 归一化：

为了防止数字爆炸（溢出）或消失（下溢），我们为下一次迭代归一化向量：

$$x_{k+1} = \frac{y}{||y||}$$

我们将从标准的 CPU 实现开始，使用 CuPy 将其移植到 GPU，并分析内存传输对性能的影响。

2. CPU 基线 (NumPy)
我们生成一个可对角化的随机密集矩阵。此数据在主机 (CPU) 上生成，并驻留在主机内存中。

实现幂迭代 (CPU)
如上所述，幂迭代算法反复将向量 $x$ 乘以矩阵 $A$（$y = Ax$）并归一化结果。我们使用一个全1向量 ($x_0$) 作为初始猜测来初始化此算法。

3. GPU 移植 (CuPy)
练习：将 CPU 实现移植到 GPU
现在轮到您了！您的任务是使用 CuPy 将 estimate_host 函数转换为在 GPU 上运行。

请记住内存空间的规则：

传输： 使用 cp.asarray() 将 A_host 从 CPU 移动到 GPU。
计算： 使用 cp 函数在 GPU 上执行数学运算。
检索： 如果我们需要打印结果或在标准 Python 中使用它，则使用 cp.asnumpy() 或 .item() 将结果移回 CPU。
提示： CuPy 尝试复制 NumPy API。在许多情况下，您只需将 np. 更改为 cp.。但是，CuPy 操作必须在设备内存中的数据上运行。

在下面的骨架代码中填写 TODO 部分：

4. 优化数据生成
在上一步中，我们在 CPU 上生成数据并将其复制到 GPU。对于大型数据集，传输时间（主机 -> 设备）可能成为瓶颈。

如果可能的话，直接在 GPU 上生成数据几乎总是更快。

练习：直接在 GPU 上生成数据
您的任务是转换 generate_host 函数，使用 CuPy 的随机函数直接在 GPU 上生成矩阵。

提示：

使用 cp.random.seed() 代替 np.random.seed()
使用 cp.random.random() 代替 np.random.random()
使用 cp.random.permutation() 代替 np.random.permutation()
使用 cp.concatenate()、cp.array()、cp.diag() 和 cp.linalg.inv()
在下面的骨架代码中填写 TODO 部分：

思考一下
两个函数都使用 seed(42)。A_host 和 A_device 是否相同？尝试比较它们：

这揭示了关于 np.random 与 cp.random 的什么信息？

5. 验证与基准测试
最后，让我们根据参考实现 (numpy.linalg.eigvals) 验证我们的准确性并对加速进行基准测试。

关于 CuPy 限制的说明： 您可能想知道为什么我们在 CPU 上使用 np.linalg.eigvals 而不是 CuPy 等效函数。原因是 CuPy 尚未实现 eigvals。虽然 CuPy 涵盖了 NumPy API 的很大一部分，但它并不支持所有函数。在假设可以直接从 NumPy 转换 CuPy 之前，请务必查看 CuPy 文档 以验证哪些函数可用。

关于验证的说明
如前所述，A_host 和 A_device 是不同的矩阵（NumPy 和 CuPy 使用不同的 RNG 实现）。然而验证通过了。为什么？

两个矩阵都构建为将一个特征值显式设置为 1.0。验证确认了幂迭代正确地找到了这个主特征值——而不是矩阵是相同的。

关键要点： 如果您需要在完全相同的数据上验证 GPU 计算与 CPU 的对比，请在一个设备上生成数据并传输到另一个设备。

使用 cupyx.profiler.benchmark 进行基准测试
我们使用 CuPy 内置的基准测试工具来进行准确的 GPU 计时。这会自动处理预热和同步。

附加题
探索更改以下参数的影响：

问题规模 (dim)： 当您增加或减少矩阵维度时，GPU 加速如何变化？尝试像 1024、2048、4096、8192 这样的值。
计算工作负载 (max_steps 和 dominance)： dominance 参数控制算法收敛的速度。较小的 dominance 意味着特征值更接近，需要更多迭代。这如何影响 CPU 与 GPU 的比较？
检查频率 (check_frequency)： 这控制我们检查收敛的频率（并通过 print 语句触发隐式 CPU 同步）。当您在每一步检查 (check_frequency=1) 与较少检查 (check_frequency=50) 时，GPU 性能会发生什么变化？
在下面进行实验：



