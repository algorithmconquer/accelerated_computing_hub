第 12 章：NVIDIA Warp 简介
概览
NVIDIA Warp 是一个开源的 Python 开发框架，专门用于开发高性能仿真和 AI 工作负载。

Warp 为编码者提供了一个清晰且极具表现力的编程模型，用于编写基于内核的 GPU 加速程序，适用于仿真 AI、机器人技术和机器学习 (ML)。

Warp 的主要特性包括：

高性能：通过结合 即时 (JIT) 编译、CUDA-X 库集成和透明的内核融合，实现与原生 CUDA C++ 代码相当的性能。
易用性：通过 Python 中的高级编程模型、内置的用于空间计算的数据结构和算法，以及对基于 Tile 的编程的支持。
支持高级仿真和 AI 工作负载：通过自动微分以及与其他 ML 和加速 Python 框架的互操作性提供支持。
本笔记本为读者提供了该库主要功能的概述。

涵盖的主题：

Warp 基础知识：数据模型、执行模型、基本语法
在 Warp 中编写基础内核
将 Warp 与其他基于 Python 的框架（如 NumPy、PyTorch 和 JAX）配合高效使用
Warp 中的自动微分
介绍
Warp 是 NVIDIA 开发的一个用于在 Python 中编写高性能仿真和图形代码的框架。Warp 的核心是基于内核的编程模型，其中 Python 函数被即时 (JIT) 编译为高效代码，可以使用 C++/CUDA 作为中间表示在 CPU 和 NVIDIA GPU 上运行。Warp 还具有反向模式自动微分系统，允许研究人员编写可微分仿真器信息，这些信息可以有选择地纳入机器学习流水线，以使用 PyTorch 或 JAX 训练神经网络。研究人员已将 Warp 应用于物理仿真、感知、机器人和几何处理等领域。

设置
Warp 是一个轻量级库，其唯一的必需依赖项是 NumPy。与许多其他加速 Python 库相比，Warp 预先打包了为 CPU 或 GPU 编译代码所需的编译器，而不需要在开发环境中安装额外的库。

大多数用户从 Python 包索引 (PyPI) 安装 Warp，在那里它可以作为 warp-lang 获取。

(此处跳过代码)

安装 warp-lang 包后，我们可以在 Python 脚本中 import warp 开始使用它。

通常，Warp 使用导入别名 wp。

我们将使用 wp.init() 显式初始化 Warp，以确保在笔记本环境中检测到 NVIDIA GPU，因为笔记本的其余部分假设至少存在一个 GPU。

(此处跳过代码)

上述单元格的输出应在 Devices: 部分下列出 "cpu" 和 "cuda:0" 设备。

用户脚本中不要求调用 wp.init()。在第一次调用需要初始化 Warp 的函数时，它将被隐式调用。

获取预构建 Warp 安装的其他方法
社区维护的 Warp Conda 包可在 conda-forge 频道获取：

bash
# 安装专门针对 CUDA Toolkit 12.6 构建的 warp-lang
$ conda install conda-forge::warp-lang=*=*cuda126*
# 安装专门针对 CUDA Toolkit 11.8 构建的 warp-lang
$ conda install conda-forge::warp-lang=*=*cuda118*
尖端（Bleeding-edge）每日构建包发布在 NVIDIA Python 包索引。由于这可以是在新版本发布之前从 main 分支获取最新功能的一种方式，但这些包尚未像 PyPI 上发布的版本那样经过彻底测试：

bash
$ pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/
为本笔记本安装额外依赖项
本入门笔记本使用了其他一些 Python 包。在继续之前，请确保这些依赖项已安装在 Python 环境中：

(此处跳过代码)

Warp 中的粒子仿真示例
我们将首先看一个基础的 Warp 程序，它求解一组粒子在重力 $g$ 和非重力 $f_n$ 影响下的运动。

这个例子旨在让我们对 Warp 程序的样子有一个整体的感觉。在随后的章节中，我们将更详细地查看基本概念。

更新方程为：

\begin{align*} a_{n} &= f_n / m + g \ v_{n+1} &= v_n + a_n \Delta t \ x_{n+1} &= x_n + v_{n+1} \Delta t. \end{align*}

我们将假设一个简单的阻力 $f_n = -b v_n$。

粒子位置和速度将被初始化为随机值。

然后程序使用半隐式欧拉积分更新每个粒子在大小为 $\Delta t$ 的每个时间步的位置和速度，共进行 100 步。在程序结束时，打印出粒子的最终位置。

(此处跳过代码)

让我们更详细地查看程序的主要部分。

定义程序常量
代码的第一部分为程序定义了一些常量：

python
num_particles = 10_000_000  # 粒子数量
num_steps = 100
mass = 0.1  # 每个粒子的质量 [kg]
g = 9.81  # 重力加速度 [m/s^2]
b = 0.05  # 阻力系数 [kg/s]
dt = 0.01 * (2 * mass / b)
gravity = wp.vec3([0.0, 0.0, -g])
除了最后一行外，所有内容都是纯 Python。在最后一行，我们看到重力被定义为 gravity = wp.vec3([0.0, 0.0, -g])。

wp.vec3 是 Warp 提供的一种内置数据类型，代表一个由三个 32 位浮点值构成的向量。

为粒子位置和速度分配数组
接下来，我们看到我们分配了一些数组来表示粒子的位置和速度：

python
# 初始位置：x, y, z 的随机值在 -1.0 到 1.0 之间
rng = np.random.default_rng(12345)
positions_np = rng.uniform(low=-1.0, high=1.0, size=(num_particles, 3))
positions = wp.array(positions_np, dtype=wp.vec3)
# 初始速度：vx, vy, vz 的随机值在 -0.5 到 0.5 m/s 之间
velocities_np = rng.uniform(low=-0.5, high=0.5, size=(num_particles, 3))
velocities = wp.array(velocities_np, dtype=wp.vec3)
Warp 中的数组是表示数据的基本方式，可以使用 wp.array() 构造函数创建。与 NumPy 类似，数组可以是多维的，并且数组的所有元素必须具有相同的数据类型。

注意，对于位置和速度，我们首先创建了包含我们要初始化粒子的数据的 NumPy 数组，然后将这些 NumPy 数组连同 wp.vec3 数据类型（与 gravity 变量类型相同）一起传递给 wp.array() 构造函数。这是 Warp 与 NumPy 互操作的常见方式之一。

代表 positions 和 velocities 数组的内存分配在我们的示例中最终位于 GPU 上，这很重要，因为我们希望使用 GPU 并行更新粒子。

定义内核
分配好粒子数据的数组后，我们随后通过编写内核 (kernel) 来定义对该数据的计算，内核本质上是一个在 GPU 上的许多线程中编译并执行的函数。实际上，Warp 中的内核也可以在 CPU 上编译和执行，但目前将限于单线程执行。

Warp 中的内核通过使用 @wp.kernel 装饰 Python 函数来定义。Warp 中的有效内核还必须遵守额外的限制，例如：

使用 Python 语言的子集
参数必须有类型
不能 return 任何内容
当内核被 启动 时，内核体将按照传递给 wp.launch() 的内核启动维度并行执行一定的次数。

相比之下，普通的 Python 函数在调用时仅执行一次。

我们将 integrate 内核定义为接受 positions 和 velocities 数组作为输入。在内核内部，我们使用 wp.tid() 获取当前线程 ID，这告诉我们这个特定线程应该处理哪个粒子（数组元素）。这允许每个线程同时处理不同的粒子。

python
@wp.kernel
def integrate(positions: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    acceleration = (-b * velocities[i]) / mass + gravity
    velocities[i] += acceleration * dt
    positions[i] += velocities[i] * dt
启动 integrate 内核
我们使用 wp.launch() 函数在 GPU 上运行我们的内核。此函数接受三个主要参数：要执行的内核函数、要启动的并行线程数量（由 dim 指定）以及与内核函数签名匹配的输入参数。

python
for step in range(num_steps):
    wp.launch(integrate, dim=(num_particles,), inputs=[positions, velocities])
integrate 内核在一个 for 循环内启动，以模拟随时间变化的粒子运动。循环的每次迭代代表一个时间步，基于物理方程更新所有粒子的位置和速度。此过程重复 num_steps 次迭代，以模拟粒子系统的全部持续时长。

打印结果
完成所有仿真时间步后，我们可以打印出最终的粒子位置：

python
print(f"Final positions: {positions}")
这个简单的 print 语句无缝工作，因为 Warp 自动处理从 GPU 内存到 CPU 内存的数据传输。在计算过程中存储在 GPU 上的 positions 数组透明地复制到 CPU，以便我们可以显示其值。

总结
在这个基础示例中，我们看到了 Warp 程序的基本构建模块：

数据管理：使用 wp.array() 在 GPU 上创建并存储数据
计算：将内核定义为装饰有 @wp.kernel 的 Python 函数以执行并行操作
执行：使用 wp.launch() 启动内核以在多个 GPU 线程中运行计算
这些组件共同实现了 GPU 上的高性能并行计算，同时保持了 Python 的易用性。在接下来的章节中，我们将更详细地探讨这些概念。

内核 (Kernels)
在 Warp 中，计算内核被定义为 Python 函数并用 @wp.kernel 装饰器标注。

定义 Warp 内核的 Python 函数必须遵守一些额外的限制，例如：

使用 Python 语言的子集
参数必须有类型
不能 return 任何内容
从概念上讲，Warp 内核与 CUDA 内核相似。当内核被 启动 时，内核体将按照通过内核启动维度指定并行执行一定的次数。

相比之下，普通的 Python 函数在调用时仅执行一次。

与 CUDA 内核一样，Warp 内核不返回值。相反，内核中的每个线程可以修改由作为内核参数传递的 Warp 数组包装的全局内存中的数据。

下面的笔记本单元格包含一个最简单的 Warp 内核示例。它使用线程索引填充一个数组。

(此处跳过代码)

内核在 GPU 上的执行
在这一点上，我们应该了解 Warp 中的内核如何映射到 GPU 上的硬件单元。

Warp 将 wp.launch() 中提供的网格（grid）维度映射到一维 CUDA 内核网格。
CUDA 网格被分解为大小均匀的单个线程块 (thread blocks)，这些线程块彼此独立执行。
Warp 默认 CUDA 网格为每块 256 个线程。虽然您可以将其调整到最大 1024，但不同的数字是否能提高性能取决于内核的工作负载和 GPU 架构。
每个线程块被分配给一个流式多处理器 (SM) 执行。
每个块可以按任何顺序调度到任何可用的 SM 上。
一旦线程块开始在 SM 上执行，它将一直在该 SM 上运行直到完成。
一个 SM 通常可以并发运行多个线程块。
SM 通过将线程块划分为硬件 Warp (hardware warps) 来处理它，每个硬件 Warp 由 32 个线程组成，这些线程以锁步方式（lockstep）执行指令（SIMT——单指令多线程）。
就使用 NVIDIA Warp 而言，这并不是一个重要的细节，但值得一提这个潜在的命名混淆源。
硬件 Warp 内的单个线程在 SM 内的 CUDA 核心上执行其指令。


图片来源：NVIDIA Developer Blog

编译流水线与内核缓存
当内核第一次启动时，模块中迄今为止定义的所有内核都将被翻译为原生 C++/CUDA 代码并进行即时编译 (just-in-time compiled)。

C++/CUDA 源码和编译后的对象都作为文件存储在 内核缓存 中，内核缓存位于 wp.config.kernel_cache_dir，这样后续启动就不必承担代码生成和编译的开销。

(此处跳过代码)

启动 fill_kernel 后，内核缓存中将出现一个 .cu 文件以及一个与 Warp 生成的 CUDA 源码和 NVIDIA 运行时编译 (NVRTC) 库的编译器输出对应的 .ptx 或 .cubin 文件。

下图显示了 Warp 编译流水线：

让我们仔细看看生成的代码。

我们将使用一个运行有限差分（finite-difference）内核的独立脚本。内核缓存将从默认位置更改，以便我们可以从侧边栏查看缓存。

(此处跳过代码)

(此处跳过代码)

探索保存在位于 Chapter_12_finite_difference_example_cache 目录下的内核缓存中的文件。

请注意，默认情况下，Warp 会为每个内核定义生成正向和反向两个版本。

正向版本是您通常习惯看到的。反向版本用于自动微分（稍后会详细介绍）。

现在，我们将对文件进行细微修改，在计算 u_out 时使用二阶有限差分。

其他一切都将保持不变（文件名、finite_difference 内核的名称、问题分辨率等）。

(此处跳过代码)

(此处跳过代码)

在上一单元格的日志中，我们应该看到类似这样的一行：

text
Module __main__ ea6bc0d load on device 'cuda:0' took 305.15 ms  (compiled)
更改 finite_difference 的内容导致模块产生了新的哈希值（例如 05beb6e）。

由于内核缓存尚未包含哈希值为 05beb6e 的 __main__ 模块的已编译代码，Python 代码被翻译为 CUDA C++ 并进行了编译。

Warp 内核内部的类型转换
因为 Warp 内核被编译为原生 C++/CUDA 代码，所以所有函数输入参数都应该是 有类型的 (typed)。

这使得 Warp 能够生成以原生速度执行的高效代码。

如果存在类型不匹配，将抛出异常，因为 Warp 不会为用户自动执行类型转换。

让我们看看如果 fill_kernel 期望一个 
int
 数组但我们给它一个 float 数组会发生什么。

(此处跳过代码)

有时您需要在 Warp 内核内部将变量强制转换为不同类型，例如将 float 与 
int
 相乘：

python
prod[i] = float(int_array[i]) * float_array[i]
或者等效地：

python
prod[i] = wp.float32(int_array[i]) * float_array[i]
作为一个练习，修复以下内核中的类型问题：

(此处跳过代码)

使用泛型创建更灵活的内核
为了方便并提高代码可重用性，Warp 支持使用 typing.Any 代替具体类型。

有关更多信息，请参阅 泛型文档。

以下示例定义了一个泛型内核，并在不同数据类型上启动了三次。

注意在类型转换中 type() 的使用。

(此处跳过代码)

Warp 内核内部的调试打印
我们可以使用 wp.printf() 从 Warp 内核内部打印 C 风格格式化字符串。

要打印向量和矩阵等复合类型，请使用 
print()
。

(此处跳过代码)

多维内核
到目前为止，我们一直在一维网格上启动内核，但我们可以使用高达四维的网格（与 Warp 数组的最大维度匹配）。

要获取多维线程索引，我们使用元组拆包：

python
i = wp.tid()
i, j = wp.tid()
i, j, k = wp.tid()
i, j, k, l = wp.tid()
以下示例在四维网格上启动一个内核，其中每个线程检索并打印其线程索引。

(此处跳过代码)

使用 
device
 关键字
device
 关键字可用于针对特定设备启动内核。

回想一下在本笔记本开始时，当我们调用 wp.init() 时，我们看到 Devices: 部分打印出了 "cpu" 和 "cuda:0" 设备。这些别名可以与 
device
 关键字一起用于启动内核和分配数组。

所有数组必须驻留在与内核启动相同的设备上。

如果我们尝试在 "cpu" 上启动带有位于 "cuda:0" 上的数组的内核，我们会得到一个错误：

(此处跳过代码)

如果一个接受 
device
 参数的 Warp API 调用（通常是数组分配或内核启动）没有提供设备，则会使用我们所谓的默认设备。

如果 Warp 检测到 NVIDIA GPU，默认设备将是 'cuda:0'，否则将是 'cpu'。

我们可以通过不带参数地调用 wp.get_device() 来获取默认设备。

(此处跳过代码)

数组 (Arrays)
内存分配通过 wp.array 类型公开。它们很重要，因为内核必须将其结果写入内存，而不是直接返回值。

数组包装了可能驻留在主机 (CPU) 或设备 (GPU) 内存中的底层内存分配。

所有数组都有一个关联的数据类型，可以是标量数据类型（例如 float、
int
）或复合数据类型（例如 vec3、matrix33）。

我们可以使用 dtype 参数设置数据类型，或者在从现有数据（例如 Python 列表或 NumPy 数组）创建数组时让 Warp 推断它。

Warp 数组当前可以使用的标量数据类型完整列表为：

名称	描述
bool	布尔型
int8	有符号字节
uint8	无符号字节
int16	有符号短整型
uint16	无符号短整型
int32	有符号整型
uint32	无符号整型
int64	有符号长整型
uint64	无符号长整型
float16	半精度浮点数
float32	单精度浮点数
float64	双精度浮点数
以下代码展示了一个规格说明不足（under-specified）的 wp.array() 的构造（形状和数据类型都是必需的）。

(此处跳过代码)

(此处跳过代码)

在实践中，通常使用以下便捷函数来构造具有基本初始化选择的 Warp 数组：

(此处跳过代码)

请注意，我们分配数组时如果不显式指定目标 
device
，它们将分配在默认设备上，如果系统检测到 GPU，则默认设备是 NVIDIA GPU。

我们可以通过检查数组的 device.is_cuda 属性来验证这一点。

(此处跳过代码)

让我们显式地在 'cpu' 设备上分配相同的数组：

(此处跳过代码)

使用 NumPy 数组初始化 Warp 数组
Warp 互操作的库之一是 NumPy。这意味着某些 Warp 函数可以接受 NumPy 数组。

正如我们在粒子仿真示例中看到的，从具有所需值的 NumPy 数组初始化 Warp 数组通常很方便。

只需将 NumPy 数组传递给 wp.array() 构造函数的第一个位置即可。

(此处跳过代码)

如果我们想在 CPU 上检索 GPU 数组（如 test_array）的值，我们需要将数据复制回 CPU 内存。

numpy() 方法对于获取 Warp 数组的临时 NumPy 视图非常有用。

如果 Warp 数组在 GPU 上，将首先在 CPU 上创建一个新数组，然后将 GPU 数组的内容复制到其中，最后返回 NumPy 视图。

如果 Warp 数组已经位于 CPU 上，则返回零拷贝（zero-copy）的 NumPy 视图。

Warp 数组的 __str__ 方法也会自动对数组调用 numpy()。

(此处跳过代码)

如果我们想在 CPU 上分配数组，我们可以显式地将参数 "cpu" 传递给 
device
 参数。

回想一下，如果不指定 
device
 参数，数组将分配在 默认设备 上，这会优先选择系统上的 GPU 而非 CPU。

(此处跳过代码)

创建多维数组
Warp 目前支持高达四维的多维数组。

(此处跳过代码)

复合类型
到目前为止，我们主要创建的是基于标量数据类型的 Warp 数组，但为了方便，也支持复合数据类型。

这里，我们分配一个数据类型为 wp.vec3 的 10 元素数组，它是一个三分量的 wp.float32 向量。我们在最初的粒子仿真示例中使用了 wp.vec3 数组。

(此处跳过代码)

在设备之间复制数组
数组中的值也可以使用 wp.copy() 直接复到另一个数组中（包括驻留在不同 GPU 上的数组之间）。

(此处跳过代码)

Python 作用域 API vs. 内核作用域 API
Warp 的某些 API 只能从 Python 作用域调用（即在 Warp 用户函数和内核外部），而其他 API 只能从 内核作用域调用（即在 Warp 内核和函数内部）。

Python 作用域 API 在 Python 参考 中有记载，而可在 Warp 内核中使用的函数则在 内置参考 中有记载。

通常，内核作用域 API 也可以在 Python 作用域中使用。这些函数在内置参考中用 Python 标签标注。

Python 语言并非所有的内容都被内核作用域支持。有些功能尚未实现，而有些功能从性能角度来看无法很好地映射到 GPU。

有关更多详细信息，请参阅 局限性 文档。

(此处跳过代码)

用户函数
用户可以使用 @wp.func 装饰器编写自己的可重用函数，内核可以调用这些函数，例如：

(此处跳过代码)

内核可以调用定义在同一模块或不同模块中的用户函数。如示例所示，用户函数的返回类型提示是可选的。

除了 wp.tid() 以外，任何可以在 Warp 内核中执行的操作都可以在用户函数中执行。

如果需要，线程索引可以通过用户函数的参数传入。

结构体 (Structs)
用户可以使用 @wp.struct 装饰器定义自己的结构。

结构可以作为内核的参数传递，例如，当需要许多参数时，可以使用它们简化内核签名。

结构也可以用做 Warp 数组的数据类型。

以下示例展示了如何为 update 内核中所需的仿真参数创建结构。

(此处跳过代码)

自动微分 (Automatic differentiation)
如果我们想在计算机程序中计算导数，我们的主要选择有：

手算导数，然后编写导数公式代码
使用有限差分近似计算导数
在像 Mathematica 这样的计算机代数系统中实现公式，然后使用符号微分求导
使用自动微分获得精确的数值导数
正如我们在前面的有限差分示例中看到的，Warp 默认会为每个内核定义生成正向和反向（伴随/adjoint）版本。

内核的反向版本可用于使用反向模式自动微分计算损失函数的梯度。

参与需要梯度的计算链的数组应在创建时附带 requires_grad=True，例如：

python
a = wp.zeros(1024, dtype=wp.vec3, requires_grad=True)
然后可以使用 wp.Tape 类记录内核启动，并重新播放它们以计算标量损失函数相对于内核输入的梯度：

python
# 正向过程
with wp.Tape() as tape:
    wp.launch(kernel=compute1, inputs=[a, b])
    wp.launch(kernel=compute2, inputs=[c, d])
    wp.launch(kernel=loss, inputs=[d, l])
# 反向过程
tape.backward(l)
反向过程完成后，相对于输入的梯度可从 array.grad 属性获取：

python
# 损失相对于输入 a 的梯度
print(a.grad)
在实践中，Warp 的自动微分功能被应用于涉及分支逻辑、循环和函数调用的更复杂算法，但我们将看一个闭式表达式（closed-form expression），因为我们可以轻松地将结果与获取数值导数的不同方法进行比较。

让我们考虑在 x = 0.5 处评估以下闭式函数。

$$ f(x) = \sin \left(x^2\right) \cdot \ln(x) + \frac{x^3}{\sqrt{1 - x^2}} $$

我们可以手算出该函数导数的解析表达式，并在 Python 中实现一个函数：

(此处跳过代码)

我们也可以使用有限差分近似来评估导数，这会带来截断误差和舍入误差：

(此处跳过代码)

随着计算量复杂度的增加，解析方法的可扩展性较差，而数值方法随输入数量增加的可扩展性也不佳（此外步长 h 的选择也很困难）。

如果我们想使用 Warp 的自动微分功能评估导数，我们将实现一个执行函数评估并将结果写入数组的内核。

(此处跳过代码)

请注意，此结果不是使用数值微分获得的。这里没有步长。

相反，程序运行了两次：

调用 wp.launch() 时，在 正向模式 下运行一次
调用 tape.backward() 时，在 反向模式 下运行一次（因为伴随项是从输出向输入反向传播的）
Warp 如何知道如何精确评估导数？ 自动微分 (AD) 系统实现了有限的一组初等运算的已知导数。链式法则被用于将初等导数组合在一起以获得整体导数。

结论
本笔记本介绍了 Warp 的核心组件。更多示例请参见 GitHub 上的 Warp 示例库。

https://github.com/shi-eric/warp-lanl-tutorial-2025-05 仓库也包含了一套 Warp 教程。

参考文献
有关 Warp 的更多信息：

"NVIDIA/warp: A Python framework for accelerated simulation, data generation and spatial computing.", GitHub, https://github.com/NVIDIA/warp, 访问日期：2025年7月2日.
Warp 开发者, "NVIDIA Warp Documentation," GitHub Pages, https://nvidia.github.io/warp/, 访问日期：2025年7月2日.
Miles Macklin, Leopold Cambier, Eric Shi, "Introducing Tile-Based Programming in Warp 1.5.0", NVIDIA Developer, https://developer.nvidia.com/blog/introducing-tile-based-programming-in-warp-1-5-0/, 访问日期：2025年7月2日.
"Warp: Differentiable Spatial Computing for Python", ACM Digital Library, https://dl.acm.org/doi/10.1145/3664475.3664543, 访问日期：2025年7月2日.
Miles Macklin, "Warp: Advancing Simulation AI with Differentiable GPU Computing in Python", NVIDIA On-Demand, https://www.nvidia.com/en-us/on-demand/session/gtc24-s63345/, 访问日期：2025年7月2日.
Miles Macklin, "Warp: A High-performance Python Framework for GPU Simulation and Graphics", NVIDIA On-Demand, https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41599/, 访问日期：2025年7月2日.
Miles Macklin, "Differentiable Physics Simulation for Learning and Robotics", NVIDIA On-Demand, https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31838/, 访问日期：2025年7月2日.
有关使用 Warp 的项目的更多信息：

"nvidia-warp · GitHub Topics", GitHub, https://github.com/topics/nvidia-warp, 访问日期：2025年7月2日.
Warp 开发者, "Publications using Warp," GitHub, https://github.com/NVIDIA/warp/blob/main/PUBLICATIONS.md, 访问日期：2025年7月2日.
有关自动微分的更多信息：

Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind, "Automatic differentiation in machine learning: a survey", The Journal of Machine Learning Research, 18(153), 1-43, 2018.
Andreas Griewank 和 Andrea Walther, "Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation", 第2版, SIAM, 2008.
Stelian Coros, Miles Macklin, Bernhard Thomaszewski, Nils Thürey, "Differentiable simulation", SA '21: SIGGRAPH Asia 2021 Courses, 1-142, 2021.
