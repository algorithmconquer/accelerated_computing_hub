第 10 章：开发者工具
CUDA Python 性能
为了在 CUDA 中实现最佳性能，您必须考虑以下几个因素：

本地化内存访问，以最大限度地减少内存延迟。
最大化每个多处理器的活动线程数，以确保硬件的高利用率。
最小化条件分支。
为了克服通过 PCIe 总线在 CPU 和 GPU 之间产生的瓶颈，我们需要：

最小化传输的数据量。大批量传输数据可以减少数据传输操作的次数。
以互补硬件架构的方式组织数据。
利用异步传输功能，允许计算和数据传输同时进行。将数据传输与计算重叠可以隐藏由数据传输引起的延迟。
Nsight Systems 和 Nsight Compute 是用于检测 CUDA 代码中瓶颈和性能缺陷的工具。

CUDA Python 正确性
CUDA 代码有时会引入编译器无法检测到的各种错误，例如：

内存访问冲突
内存泄漏
数据竞争条件
API 使用不当
这些错误可能导致程序行为异常、崩溃或性能下降。Compute Sanitizer 是 NVIDIA 提供的一套运行时错误检测工具，可帮助开发者识别和调试 CUDA 应用程序中的此类问题。

常见陷阱
最常见的错误是在 GPU 节点上运行仅限 CPU 的代码。只有显式编写为在 GPU 上运行的代码才能利用 GPU。请确保您的代码使用了正确的 GPU 加速库、驱动程序和硬件。

GPU 利用率为零 检查并确保您的软件已启用 GPU 支持。只有明确编写为使用 GPU 的代码才能利用它们。 确保您的软件环境配置正确。在某些情况下，运行 GPU 代码需要特定的库。检查您的依赖项、CUDA 工具包版本以及软件环境要求。

GPU 利用率低（例如低于 ~15%） 使用了超过必要的 GPU。您可以通过进行扩展性分析（scaling analysis）来找到 GPU 数量和 CPU 核心数的最优组合。 检查流程的吞吐量。如果您正在向慢速内存写入输出、进行不必要的拷贝或频繁在 CPU 和 GPU 之间切换，可能会看到利用率较低。

内存错误

访问冲突错误。读取或写入不允许或未获授权的内存位置可能导致不可预测的行为和系统崩溃。
内存泄漏。当内存已分配但未正确释放时，应用程序将消耗 GPU 显存资源但并不利用它们。已分配的内存将无法用于进一步的计算。
开始使用 CUDA Python 开发者工具
前提条件
本文档中的步骤假设用户拥有能够在 GPU 上运行 CuPy 和 Numba 代码的环境。请参阅这些各自的项目进行设置。

Nsight Systems（也可在 CUDA Toolkit 中获得）
Nsight Compute（也可在 CUDA Toolkit 中获得）
Compute Sanitizer（也可在 CUDA Toolkit 中获得）
nvtx Python 绑定
使用 Nsight Systems 进行分析
Nsight Systems 是一款平台分析工具，旨在为用户提供整个平台性能活动的宏观、时间相关的视图。这包括 CPU、GPU、内存、网络、操作系统和应用程序层级的指标。它有助于识别最大的优化机会，并进行调优以跨所有可用资源高效扩展。本教程仅触及 Nsight Systems 功能的皮毛。完整的详细信息请参见文档。

使用 Nsight Systems GUI 设置分析任务
打开 Nsight Systems GUI 后，选择要分析的目标机器。这可以是本地机器或远程服务器。本例使用本地目标。要通过 Nsight Systems 分析 Python 工作负载，请在“Command line with arguments:”字段中指向 Python 解释器和要运行的 Python 文件（包括任何参数）。确保 Python 可执行文件处于拥有应用程序所需所有依赖项的环境中。例如：“C:\Users\myusername\AppData\Local\miniconda3\python.exe C:\Users\myusername\cupyTests\cupyProfilingStep1.py <如果需要，添加参数>”

同时填写 Python 可执行文件运行的“Working directory”。

推荐设置/标志

分析 Python 的一组良好初始标志包括：

收集 CPU 上下文切换追踪 (Collect CPU context switch trace)
收集 CUDA 追踪 (Collect CUDA trace)
收集 GPU 指标 (Collect GPU metrics)
Python 分析选项：
收集 Python 回溯样本 (Collect Python backtrace samples)
您可以在此处了解有关所有选项的更多信息。

CuPy 分析示例
在此示例中，我们创建了两个 CuPy 数组。然后对其中一个进行排序并计算点积。

python
import sys
import cupy as cp
def create_array(x, y):
    return cp.random.random((x, y), dtype=cp.float32)
def sort_array(a):
    return cp.sort(a)
def run_program():
    print("初始化步骤...")
    arr1 = create_array(10_000, 10_000)
    arr2 = create_array(10_000, 10_000)
    print("排序步骤...")
    arr1 = sort_array(arr1)
    print("点积步骤...")
    arr3 = cp.dot(arr1, arr2)
    
    print("完成")
    return
if __name__ == '__main__':
    run_program()
第 1 步 - 分析 CuPy 工作负载

首先，使用上述设置和标志对该 CuPy 示例运行初始分析。如果无法从 GUI 启动分析任务，也可以通过命令行启动。运行此分析的 CLI 命令示例如下。某些标志可能会根据您的具体设置而有所不同。

nsys profile --gpu-metrics-device=all --python-sampling=true --python-sampling-frequency=1000 --trace=cuda --cpuctxsw=process-tree python "/home/myusername/cupytest1.py"

分析完成后，在时间轴的 Processes 行下找到 Python 进程线程。通过左键单击并拖动感兴趣的区域来选择并放大 Python 线程的活动部分。然后右键单击选择“Zoom into selection”。如果您将鼠标悬停在 Python Backtrace 行中的样本上，将弹出一个窗口，显示抽取样本执行时的调用栈。



CuPy 在执行时会在底层调用 CUDA 内核。Nsight Systems 会自动检测这些内核。展开 CUDA HW 行以查看内核的调度情况。



查看 GPU Metrics > GPU Active 和 SM Instructions 行来验证 GPU 是否正在被使用。您可以将鼠标悬停在该行的一个点上以查看利用率百分比。



第 2 步 - 添加 nvtx

Nsight Systems 可以自动检测 CUDA 内核以及来自许多其他框架或库的 API。此外，nvtx 标注模块使用户能够标记自己的应用程序，以便在时间轴上查看自定义的追踪事件和范围。nvtx Python 模块 通过 pip 提供，可以使用以下命令安装：

pip install nvtx

下面的代码将 nvtx 添加到 CuPy 应用程序中，并在工作负载的各个阶段定义了带颜色的范围。运行此新版本的分析任务以在时间轴上查看 nvtx 情况。如果使用 CLI，请将标志更新为 "--trace=nvtx,cuda"

python
import sys
import cupy as cp
import nvtx
def create_array(x, y):
    return cp.random.random((x, y), dtype=cp.float32)
def sort_array(a):
    return cp.sort(a)
def run_program():
    print("初始化步骤...")
    nvtx.push_range("init_step", color='green')
    arr1 = create_array(10_000, 10_000)
    arr2 = create_array(10_000, 10_000)
    nvtx.pop_range()
    print("排序步骤...")
    nvtx.push_range("sort_step", color='yellow')
    arr1 = sort_array(arr1)
    nvtx.pop_range()
    nvtx.push_range("dot_step", color='magenta')
    print("点积步骤...")
    arr3 = cp.dot(arr1, arr2)
    nvtx.pop_range()
    
    print("完成")
    return
if __name__ == '__main__':
    nvtx.push_range("run_program", color='white')
    run_program()
    nvtx.pop_range()
Python 进程 CPU 线程的 NVTX 行显示了 CPU 何时处于这些范围内。CUDA HW 栏下的 NVTX 行显示了这些范围在 GPU 上处于活跃状态的时间。由于 GPU 执行调度的原因，注意它们并没有完全对齐。您还可以看到 CUDA 内核如何映射到代表工作负载阶段的各个 nvtx 范围。

在这个特定的例子中，我们可以在 GPU Metrics > SM Instructions > Tensor Active 行中看到，在内核运行期间，GPU 上的张量核心（Tensor cores）并不活跃。张量核心可以为计算密集型内核增加大量性能。下一步将是激活它们。



第 3 步 - 启用张量核心 (Tensor cores)

CuPy 文档描述了如何通过环境变量启用张量核心。在该版本中添加了以下行：

os.environ["CUPY_TF32"] = "1"
运行另一个 Nsight Systems 分析任务，以查看此版本中张量核心的活动情况。

python
import sys
import cupy as cp
import nvtx
import os
def create_array(x, y):
    return cp.random.random((x, y), dtype=cp.float32)
def sort_array(a):
    return cp.sort(a)
def run_program():
    print("初始化步骤...")
    nvtx.push_range("init_step", color='green')
    arr1 = create_array(10_000, 10_000)
    arr2 = create_array(10_000, 10_000)
    nvtx.pop_range()
    print("排序步骤...")
    nvtx.push_range("sort_step", color='yellow')
    arr1 = sort_array(arr1)
    nvtx.pop_range()
    nvtx.push_range("dot_step", color='magenta')
    print("点积步骤...")
    arr3 = cp.dot(arr1, arr2)
    nvtx.pop_range()
    
    print("完成")
    return
if __name__ == '__main__':
    os.environ["CUPY_TF32"] = "1"
    nvtx.push_range("run_program", color='white')
    run_program()
    nvtx.pop_range()


注意，现在在点积期间正在使用张量核心，并且点积范围在 GPU 上的运行时间更短了：312ms -> 116ms。

第 4 步 - 使用标注文件

Nsight Systems 还可以通过标注文件自动追踪 Python 模块（在此例中为 CuPy）中的特定函数。本例指向文件“cupy_annotations.json”，内容如下：

json
[
    {
        "_comment": "CuPy Annotations",
        "module": "cupy",
        "color": "black",
        "functions": ["random.random", "dot", "sort"]
    }
]
该 JSON 对象指示模块“cupy”中的函数“random.random”、“dot”和“sort”应被追踪，并在时间轴上显示为黑色范围。如下所示，将此文件添加到配置中的“Python Functions trace”字段。



要通过 CLI 执行此操作，请添加类似 "--python-functions-trace="/home/myusername/cupy_annotations.json" " 的标志。运行另一个分析任务以查看自动追踪。



Numba 分析示例
虽然 Nsight Systems 显示了平台范围的分析信息和一些 GPU 特定数据（如 GPU 指标），但它并不深入探讨 GPU 内核本身。这就是 Nsight Compute 的用武之地。Nsight Compute 在内核于 GPU 上运行时对其进行详细的性能分析。历史上，内核都是用 C 等原生语言编写的，但像 Numba 这样的新技术使 Python 开发者也能编写内核。本节将介绍如何使用 Nsight Compute 分析 Numba 内核。有关 Nsight Compute 的更多细节，请查看 Nsight Compute 文档。

使用 Nsight Compute GUI 设置分析任务

要使用 Nsight Compute 分析 Numba 应用程序，请从 GUI 打开“Connect”对话框。选择 Python 解释器二进制文件作为“Application Executable”。确保此解释器在拥有应用程序所需所有依赖项（例如支持 Numba 的 Conda 环境）的环境中运行。然后填写“Working Directory”字段，并在“Command Line Arguments”字段中放入您的 Python 文件和任何附加命令行参数。这告诉 Nsight Compute 如何启动您的工作负载进行分析。



推荐设置/标志

Nsight Compute 有很多选项可以配置分析任务。本指南不打算涵盖所有选项，但在文档中有大量附加信息。Numba 分析的一个良好起点是选择 Profile 活动。在 Filter > Kernel Base Name 下拉菜单中选择 “Demangled”。在 Other > Enable CPU Call Stack 中选择 Yes，并将 Other > CPU Call Stack Types 选择为 All 或 Python。

Metrics 选项卡是您选择要收集哪些性能指标的地方。指标被拆分为多个集合，“detailed”集合是一个很好的起点。您可以在 内核分析指南 中了解有关指标的更多信息。更新这些设置后，点击 Launch 开始自动分析。Nsight Compute 将通过多通行回放（multi-pass replay）机制分析它遇到的每个内核，并在完成后报告结果。如果无法从 GUI 进行分析，您可以在 GUI 中配置分析任务，并从 Common 选项卡中的 "Command Line:" 复制相应的命令。此配置的一个示例命令可能是：

ncu --config-file off --export "\home\myusername\r%i" --force-overwrite --launch-count 3 --set detailed --call-stack --call-stack-type native --call-stack-type python --nvtx --import-source yes \home\myusername\numbaTest1.py

Nsight Compute 分析示例演练
在这个简单的例子中，有一个做向量加法的 Numba 内核。它接收三个向量，将其中两个相加，并将和存入第三个向量。请注意，"@cuda.jit" 装饰器带有参数 "(lineinfo=True)"。这对于将内核性能数据解析回源代码行非常重要。按照上述设置，启动分析任务以查看内核性能。

python
import numpy as np
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
@cuda.jit(lineinfo=True)
def vecadd(a, b, c):
    tid = cuda.grid(1)
    size = len(c)
    if tid < size:
        c[tid] = a[tid] + b[tid]
def run_program():
    np.random.seed(1)
    N = 500000
    a = cuda.to_device(np.random.random(N))
    b = cuda.to_device(np.random.random(N))
    # a = cuda.to_device(np.float32(np.random.random(N)))
    # b = cuda.to_device(np.float32(np.random.random(N)))
    c = cuda.device_array_like(a)
    vecadd.forall(len(a))(a, b, c)
    print(c.copy_to_host())
分析完成后，Summary 页面会显示分析过的内核概览。在本例中只有一个。展开 “Demangled Name” 列可以看到这就是我们用 Numba 编写的 “vecadd” 内核。Summary 包含一些基本信息，包括内核时长、计算和内存吞吐量。它还列出了被触发的主要性能规则，以及修正这些问题后的估计提速值。



双击内核图标将打开包含更多信息的 Details 页面。

顶部的 “GPU Speed of Light Throughput” 部分显示此内核的 Memory（内存）使用率远高于 Compute（计算）。Memory Workload Analysis 部分显示了发往设备内存的显著流量。



Compute Workload Analysis 部分显示大部分计算正在使用 FP64 流水线。



底部的 Source Counters 部分显示了停顿（stalls）最多的源代码位置，点击其中一个会打开 Source 页面。



由于这是一个非常简单的内核，大多数停顿都发生在加法语句上，但对于更复杂的内核，这种详细程度将非常宝贵。此外，Context 页面将显示导致此内核执行的 CPU 调用栈。



在这个例子中，我们没有在 NumPy 中指定数据类型，因此默认为 FP64。这导致了意料之外的内存流量增加。要手动切换到使用 FP32 数据类型，请将以下行：

python
a = cuda.to_device(np.random.random(N))
b = cuda.to_device(np.random.random(N))
改为：

python
a = cuda.to_device(np.float32(np.random.random(N)))
b = cuda.to_device(np.float32(np.random.random(N)))
切换到 FP32 数据类型并重新运行分析后，我们可以看到内核运行时大幅下降，内存流量也随之减少。将初始结果设置为 Baseline（基准线）并打开新结果，将自动对两者进行比较。注意到 FP64 的使用已经消失，内核从 59us 加速到了 33us。



Nsight Compute 拥有海量的性能数据和内置专业指南。Details 页面上的每个部分都包含特定指标类别的详细信息，包括向导式分析规则和描述。了解所有这些功能的最佳方法是在您的工作负载上进行尝试，并利用文档和相关资料进行辅助。

使用 Compute Sanitizer 检查 CUDA Python 正确性
Compute Sanitizer 是一套用于检测代码错误的命令行工具。可用的工具包括：

Memcheck (默认)：检测内存访问错误，如越界访问和非对齐内存访问。
Racecheck：识别共享内存中可能导致非确定性行为的数据竞争。
Initcheck：查找可能导致未定义行为的未初始化内存访问。
Synccheck：检测可能导致死锁或竞争条件的无效同步模式。
要选择应使用的工具，请带 "--tool" 选项运行 Compute Sanitizer，如下所示：

compute-sanitizer --tool <memcheck|racecheck|synccheck|initcheck> python <python_app.py>

您可以在此处找到有关如何使用该工具的更多信息。基本上，首先不带任何参数运行它是一个好主意，这将触发 Memcheck。Memcheck 工具将提供检测到的内存访问错误列表，以及如下例所示的 Python 回溯信息。

Compute Sanitizer Numba 示例
python
# 文件: main.py
import numpy as np
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
@cuda.jit('void(int32[:], int32[:])', lineinfo=True)
def invalid_read_kernel(x, out):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    if pos < x.size:
        out[pos] = x[pos + 2]  # 越界访问
def launchKernel():
    invalid_read_kernel[blockspergrid, threadsperblock](<d_x, d_out>)
# 初始化数据
n = 100
x = np.arange(n).astype(np.int32)
out = np.empty_like(x)
# 将数据传输到设备
d_x = cuda.to_device(x)
d_out = cuda.to_device(out)
# 为任务设置足够的线程
threadsperblock = 32
blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
# 运行内核
launchKernel()
# 同步设备
cuda.synchronize()
# 将结果拷贝回主机
out = d_out.copy_to_host()
print(out)
上面的 Numba 代码包含从数组 x 中越界读取的操作。在内核运行期间，由于索引错误，invalid_read_kernel 可能会尝试访问 x 范围之外的内存。运行以下命令进行清理：

compute-sanitizer python main.py

将得到以下输出：



您可以看到 Compute Sanitizer 正确地识别了失败的内核运行，提供了有关问题的详细信息，并输出了主机 Python 回溯和设备回溯信息。

注意：在 @cuda.jit 装饰器中需要 'lineinfo=True' 选项，以便在设备位置行和设备回溯中显示行号。

Compute Sanitizer Numba 和 ctypes 示例
对于使用 ctypes 调用已编译 CUDA 库函数的 Numba 代码，Compute Sanitizer 同样可以正常工作。如下例所示，它可以准确地连接来自 Python 和 CUDA 组件的主机回溯信息。

cuda
// 文件: cuda_code.cu
#include <stdio.h>
#if defined(_WIN32) || defined(WIN32)
#define EXPORT_FN __declspec(dllexport)
#else
#define EXPORT_FN
#endif
extern "C"
__global__ void invalid_read_kernel(int *x, int *out, int n) {
    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos < n) {
        out[pos] = x[pos+2]; // 越界访问
    }
}
extern "C" 
void launch_kernel(int *x, int *out, int n, int threadsperblock) {
    printf("正在启动 CUDA 内核...\n");
    int blockspergrid = (n + (threadsperblock - 1)) / threadsperblock;
    invalid_read_kernel<<<blockspergrid, threadsperblock>>>(x, out, n);
}
extern "C" 
EXPORT_FN void do_stuff(int *x, int *out, int n, int threadsperblock) {
    printf("正在执行操作...\n");
    launch_kernel(x, out, n, threadsperblock);
}
python
# 文件: main.py
import os
import numpy as np
import ctypes
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
def run_lib_func():
    # 加载共享库
    if os.name == 'nt':  # Windows
        print("运行在 Windows 上")
        lib = ctypes.CDLL('./cuda_code.dll')
    elif os.name == 'posix':  # Linux 或类 Unix
        print("运行在 Linux 或 Unix 上")
        lib = ctypes.CDLL('./libcuda_code.so')
    else:
        print("未知操作系统")
        exit()
    # 初始化数据
    n = 100
    x = np.arange(n).astype(np.int32)
    out = np.empty_like(x)
    # 在设备上分配内存
    x_gpu = cuda.to_device(x)
    out_gpu = cuda.to_device(out)
    # 为任务设置足够的线程
    threadsperblock = 32
    # 获取设备指针
    x_gpu_ptr = ctypes.c_void_p(int(x_gpu.device_ctypes_pointer.value))
    out_gpu_ptr = ctypes.c_void_p(int(out_gpu.device_ctypes_pointer.value))
    # 运行内核
    lib.do_stuff(x_gpu_ptr, out_gpu_ptr, ctypes.c_int(n), ctypes.c_int(threadsperblock))
    # 同步设备
    cuda.synchronize()
    # 将结果拷贝回主机
    out = out_gpu.copy_to_host()
    print(out)
run_lib_func()
使用以下命令运行 Compute Sanitizer：

compute-sanitizer python main.py

将得到以下输出：