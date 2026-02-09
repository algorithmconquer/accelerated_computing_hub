第 11 章：使用 cuPyNumeric 进行分布式计算
简介
cuPyNumeric 是一个兼容 NumPy 的库，专为大规模计算而设计。它是 NumPy API 的分布式和加速替代方案，这意味着在笔记本电脑上运行的相同代码可以扩展到多节点、多 GPU 执行。cuPyNumeric 构建在 Legate 之上，Legate 通过利用 Legion 编程模型 提供可扩展且易于访问的 GPU 加速和分布式计算能力。Legate 和像 cuPyNumeric 这样的库的目标是通过简化对大型 CPU 和 GPU 集群的处理，使高性能计算大众化。

cuPyNumeric vs NumPy
cuPyNumeric 旨在拥有 NumPy 的所有特性，使 NumPy 用户可以轻松地在其自己的问题上尝试 cuPyNumeric。然而，存在几个关键区别：

可扩展性：cuPyNumeric 能够在大型 CPU/GPU 集群上运行，使其更适合大规模数据处理，例如与新兴 AI 工作负载相关的任务。
GPU 加速：由于 cuPyNumeric 利用了 GPU 加速，与局限于 CPU 执行的 NumPy 相比，它可以提供显著的性能提升。通过利用更快的 GPU，cuPyNumeric 可以显著简化更快处理大量数据的任务。
分布式计算：cuPyNumeric 支持分布式计算，允许将计算分布在多个节点上。这使用户能够更高效地处理大型问题，与单独使用 NumPy 相比，减少了总计算时间。
硬件要求
单节点设置
建议要求：

GPU: NVIDIA V100 或更高版本
CPU: AMD EPYC 或 Intel Xeon（支持多核，至少 16 核）
系统内存: 64 GB 或更多（取决于数据集大小）
其他选项
cuPyNumeric 也可以在 HPC 集群上运行，例如位于国家能源研究科学计算中心（NERSC）的大型 Perlmutter 超级计算机。

基础 Linux 安装
首先，请按照以下步骤在您的计算机上下载 CUDA：

访问 CUDA 工具包下载。
选择您的操作系统、架构、发行版和版本。
选择安装程序类型（例如 runfile 或 deb）。
按照针对您所选选项提供的特定安装说明进行操作。
接下来，使用 pip 或 conda 安装 cuPyNumeric。

Pip
Python 轮子包（wheel packages）适用于 Linux（x86 和 aarch64）上的 Python 3.10–3.12，可从 PyPI 获取。要将这些安装到现有环境中，请运行以下命令：

sh
pip install nvidia-cupynumeric
或者创建一个新环境：

sh
python -m venv myenv
source myenv/bin/activate
pip install nvidia-cupynumeric
这将安装最新版本的 cuPyNumeric 和相应版本的 Legate。

Conda
首先，通过在终端中粘贴以下命令在计算机上安装 Conda：

sh
mkdir -p ~/miniforge3 
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge3/miniforge.sh
bash ~/miniforge3/miniforge.sh -b -u -p ~/miniforge3 
rm -rf ~/miniforge3/miniforge.sh 
~/miniforge3/bin/conda init bash
~/miniforge3/bin/conda init zsh
source ~/.bashrc
接下来，同样通过在终端中执行这些命令来安装 Legate 和 cuPyNumeric：

sh
conda create -n legate -c conda-forge -c legate cupynumeric
conda activate legate
HPC 集群安装
在像 Perlmutter 这样的 HPC 集群上安装需要额外的步骤，以便为系统 MPI 和 GASNet 库构建包装器。有关完整详细信息，请参阅 Legate 文档中的基于 GASNet 的安装。

运行 cuPyNumeric 代码
我们将首先演示一个简单的点积示例。创建一个新文件 main.py，其中包含以下代码：
from legate.timing import time
import cupynumeric as np
# 定义向量的大小
size = 100000000 
start_time = time()
# 生成两个指定大小的随机向量
vector1 = np.random.rand(size)
vector2 = np.random.rand(size)
# 使用 cuPyNumeric 计算点积
dot_product = np.dot(vector1, vector2)
end_time = time()
elapsed_time = (end_time - start_time)/1000
print("点积结果:", dot_product)
print(f"点积耗时 {elapsed_time:.4f} 毫秒")
单节点
要在具有两个 GPU 的单台工作站上运行上述代码，执行以下命令：

sh
legate --gpus 2 ./main.py
多节点 HPC
在 Perlmutter 等系统上运行示例之前，需创建一个交互式会话。例如：

sh
salloc --nodes 2 --ntasks-per-node 1 --qos interactive --time 01:30:00 --constraint gpu --gpus-per-node 4 --account=<acct_name>
此命令在 2 个 GPU 节点上创建了一个分配，每个节点有 4 个 GPU。确保已激活 legate 环境。如果未激活，请加载 Conda 模块并激活环境。现在，运行示例程序：

sh
legate --gpus 4 --launcher srun --nodes 2 ./main.py
此调用通过 legate 驱动程序运行 legate，并执行 main.py 示例。通常，可以通过简单加载相应模块在 Python 中启动 Legate，但在多节点设置中，需要使用驱动程序来启用 --launcher 和相关选项。要查看生成的调用的更多详细信息，请在 ./main.py 之前添加 --verbose 选项。有关命令行界面选项的更多信息，请参阅以下部分，或访问此链接并转到“CLI Options”部分：Legate CLI 选项。

在 Perlmutter 上运行上述代码会产生如下输出：

点积结果: 25001932.012924932
点积耗时 141.2350 毫秒
点积结果: 25001932.012924932
点积耗时 141.2350 毫秒
请注意有两个重复的输出。默认情况下，legate 会输出所有 rank 的所有输出，因此结果将重复，次数恰好等于 Legate rank 的数量（上述示例中为 2）。设置环境变量 LEGATE_LIMIT_STDOUT=1 后，只能看到第一个 rank 的输出。

示例：矩阵乘法
矩阵乘法是一种基本操作，将两个矩阵组合以产生一个新矩阵。新矩阵的元素是通过计算第一个矩阵的相应行和第二个矩阵的相应列的乘积之和得到的。在扩展矩阵乘法时，必须考虑负载均衡和并行化等多种因素。随着工作分布在不同设备中，处理矩阵的部分和变得至关重要，并可能需要更大规模的归约（reduction）。当使用 cuPyNumeric 时，所有这些基础方面都由库处理，用户可以专注于高级算法。

如果用户不使用 cuPyNumeric，他们在编写代码时将不得不显式地担心输入分发、收集和同步等事项。下图描述了数据并行中常见的一种典型的散播-收集（scatter-gather）方法。

(注意：下图并未描绘 cuPyNumeric 的并行方法。相反，它说明了在不使用 cuPyNumeric 的情况下可能如何处理并行性。)
