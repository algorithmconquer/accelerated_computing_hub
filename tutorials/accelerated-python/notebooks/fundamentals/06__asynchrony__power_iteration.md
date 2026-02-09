异步性与幂迭代
目录
简介与设置
1.1 环境设置
理论：流与同步
基线实现
分析基线
使用 NVTX 提高可见性
实现异步性
性能分析
1. 简介与设置
GPU 编程本质上是异步的。在本练习中，我们将探讨使用 CuPy 时这种行为的含义，并学习如何使用分析工具分析执行流程。

我们将重新审视幂迭代算法。我们的目标是采用标准实现，对其进行分析以识别由隐式同步引起的瓶颈，然后使用 CUDA 流和异步内存传输对其进行优化。

1.1 环境设置
首先，我们需要确保 Nsight Systems 分析器 (nsys)、Nsightful 和 NVTX 已安装且可用。

2. 理论：流与同步
所有 GPU 工作都是在流上异步启动的。流中的工作项按顺序执行。如果您在一个流上启动 f，稍后在该同一流上启动 g，则 f 将在 g 之前执行。但是，如果 f 和 g 在不同的流上启动，则它们的执行可能会重叠。

CuPy 如何处理这些：

默认流： 除非另有指定，CuPy 在默认 CUDA 流上启动工作。
顺序设备执行： 默认情况下，CuPy 工作在 GPU 上顺序执行。
异步主机执行： 从 Python（主机）的角度来看，代码通常在启动 GPU 内核后立即返回，而无需等待工作实际完成。
TODO: 即使 CuPy 是异步的，某些操作也会强制 CPU 等待 GPU 完成。您认为哪些操作会隐式同步主机和设备？

3. 基线实现
我们将从幂迭代算法的基线实现开始。

注意： 下面的单元格将代码写入名为 power_iteration__baseline.py 的文件。我们这样做是因为我们必须通过命令行使用 Nsight Systems 分析器运行代码。

4. 分析基线
现在让我们通过在 Nsight Systems nsys 工具下运行代码来对其进行分析。其语法为 nsys <nsys 标志> <您的程序> <您的程序参数>。它将在运行您的程序时收集程序中所有正在进行的操作的鸟瞰图。

现在让我们查看我们的报告并探索程序中发生了什么。

TODO: 运行下一个单元格，它将生成报告并创建一个按钮，点击后将在 Perfetto（一个基于 Web 的无需安装的可视化分析器）中打开它。

附加题： 下载 Nsight Systems GUI 并在其中打开报告以查看更多信息。

5. 使用 NVTX 提高可见性
Nsight Systems 向我们展示了大量信息——有时太多了，而且并非全部相关。

有两种方法可以过滤和注释我们在 Nsight Systems 中看到的内容。

第一种是限制我们在程序中开始和停止分析的时间。在 Python 中，我们可以使用 cupyx.profiler.profile() 来做到这一点，它为我们提供了一个 Python 上下文管理器。在其作用域内使用的任何 CUDA 代码都将包含在分析中。

not_in_the profile()
with cpx.profiler.profile():
  in_the_profile()
not_in_the_profile()
为了使其工作，我们必须将 --capture-range=cudaProfilerApi --capture-range-end=stop 作为标志传递给 nsys。

我们也可以注释代码的特定区域，这些区域将显示在分析器中。我们甚至可以为这些区域添加类别、域和颜色，并且它们可以嵌套。要添加这些注释，我们使用 nvtx.annotate()，这是另一个 Python 上下文管理器，来自一个名为 NVTX 的库。

with nvtx.annotate("Loop")
  for i in range(20):
     with nvtx.annotate(f"Step {i}"):
       pass
TODO: 返回到前面的单元格，通过添加以下内容来改进分析结果：

nvtx.annotate() 区域。请记住，您可以嵌套它们。
在运行求解器的 start =/stop = 行周围添加一个 cpx.profiler.profile()。
将 --capture-range=cudaProfilerApi --capture-range-end=stop 添加到 nsys 标志中。
然后，捕获另一个分析并查看您是否可以确定如何改进代码。具体来说，考虑一下我们如何添加更多异步性。

6. 实现异步性
记住我们学过的关于流以及如何在 CuPy 中使用它们的知识：

默认情况下，单个线程内的所有 CuPy 操作都在同一流上运行。您可以使用 cp.cuda.get_current_stream() 访问此流。
您可以使用 cp.cuda.Stream(non_blocking=True) 创建一个新流。使用 with 语句将流用于块内的所有 CuPy 操作。
您可以通过在流上调用 .record() 来在流上记录事件。
您可以通过在事件（或整个流）上调用 .synchronize() 来进行同步。
默认情况下，内存传输会阻塞。您可以使用 cp.asarray(..., blocking=False)（用于主机到设备的传输）和 cp.asnumpy(..., blocking=False)（用于设备到主机的传输）异步启动它们。
TODO: 将您的 NVTX 和 CuPy 分析器区域的内核从之前的单元格复制到下面的单元格中。然后，尝试通过添加异步性来提高性能。确保不要复制粘贴 %%writefile 指令。

现在让我们确保它能工作：

7. 性能分析
在我们分析改进后的代码之前，让我们比较一下两者的执行时间。

接下来，让我们捕获改进后代码的分析报告。

最后，让我们在 Perfetto 中查看分析报告，并确认我们已经消除了空闲时间。