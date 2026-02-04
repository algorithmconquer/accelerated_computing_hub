第 3 章：GPU 上的 Python
编译计算 vs. 解释计算
Python 是科学、工程、数据分析和深度学习领域最受欢迎的编程语言之一。然而，由于它是一种解释型语言，一直被认为对于高性能计算来说太慢了。

编译成机器码的程序比解释型语言具有速度优势，因为在指令在处理器上执行之前不需要中间步骤。

此外，由于解释器一次只能执行一个线程，Python 线程的行为与操作系统中的线程不同。全球解释器锁（GIL）本质上确保了线程不能并行操作，导致 Python 中的线程表现得像单核 CPU。通常，Python 程序员必须依靠并发程序来利用并行性，而不是多线程。

Python 中的许多外部库已经使用 C 或 C++ 等其他语言实现，以利用多线程。其中一个库是 NumPy，它是 Python 数值计算的基石。

Python 科学库概览
Python 科学计算生态系统中的 NumPy
NumPy（Numerical Python 的简称）于 2005 年通过将 Numarray 合并到 Numeric 中而创建。从那以后，开源的 NumPy 库已经演变成 Python 科学计算的核心库。它已成为许多其他科学库（如 SciPy、Scikit-learn、Pandas 等）的基础。NumPy 对科学界如此具有吸引力的主要原因是它提供了一个便捷的 Python 接口，可以高效地处理多维数组数据结构；NumPy 数组数据结构也称为 ndarray，即 n 维数组（n-dimensional array）的简称。

除了大部分采用 C 语言实现并使用 Python 作为“胶水语言”外，NumPy 在数值计算方面如此高效的主要原因是 NumPy 数组使用连续的内存块，这些内存块可以被 CPU 高效缓存。相比之下，Python 列表是包含指向内存中随机位置对象的指针数组，不容易被缓存，且内存查找开销更高。然而，这种计算效率和低内存占用是有代价的：NumPy 数组的大小是固定的且是同质的（homogeneous），这意味着所有元素必须具有相同的类型。同质 ndarray 对象的优势在于 NumPy 可以使用高效的 C 代码执行操作，避免了 Python API 昂贵的类型检查和其他开销。虽然在 Python 列表末尾添加和删除元素非常高效，但改变 NumPy 数组的大小却非常昂贵，因为这需要创建一个新数组并将旧数组的内容搬运到想要扩展或收缩的新数组中。

除了在数值计算方面比原生 Python 代码更高效外，由于支持向量化操作（vectorized operations）和广播（broadcasting），NumPy 代码也更加优雅和易读。我们将在本文中探索这些功能。



今天，NumPy 构成了科学 Python 计算生态系统的基础（见 numpy.org）。虽然已经有一些基础库的 GPU 加速版本正在开发中，但所有科学库都有机会利用这些优化。



在 Python 科学计算库中使用 CuPy
cupy.ndarray 类是 CuPy 的核心，它是 NumPy numpy.ndarray 的替代类。这使得它成为进入 Python CUDA 生态系统的绝佳切入点，特别是对于那些基于对 NumPy 的依赖构建软件的科学库开发人员而言。

CuPy 项目的目标是为 Python 用户提供 GPU 加速能力，而无需深入了解底层的 GPU 技术。CuPy 团队致力于提供：

完整的 NumPy 和 SciPy API 覆盖，以实现全面的无缝替换，以及用于最大化性能的高级 CUDA 功能。
成熟且高质量的库，作为所有需要加速的项目（从实验室环境到大规模集群）的基础包。
CuPy 构建在 CUDA 之上，并提供了更高层次的抽象，使得在不同的 GPU 架构和 CUDA 版本之间移植代码变得更加容易。这意味着 CuPy 代码可以潜在地在不同的兼容 CUDA 的系统上运行，而无需进行重大修改。

在底层，CuPy 利用了 CUDA 工具包库（包括 cuBLAS、cuRAND、cuSOLVER、cuSPARSE、cuFFT、cuDNN 和 NCCL），以充分利用 GPU 架构。

资源
NumPy: https://numpy.org/

CuPy: https://cupy.dev/

NumPy 和 CuPy 之间的区别: https://docs.cupy.dev/en/stable/user_guide/difference.html

