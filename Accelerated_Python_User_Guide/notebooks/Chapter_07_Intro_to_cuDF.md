第 7 章：使用 cuDF 处理数据帧
cuDF 是一个用于 Python GPU 加速计算的数据帧（DataFrame）库。cuDF 提供了类似 pandas 的 API，数据工程师和数据科学家对此会感到非常熟悉，因此他们可以轻松地使用它来加速工作流程，而无需深入了解 CUDA 编程的细节。

cuDF 是 NVIDIA RAPIDS 库套件的一部分，该套件包含 GPU 加速的数据科学和 AI 库，其 API 与最流行的开源数据工具相匹配。它们的作用是作为第 3 章“GPU 上的 Python”中详细介绍的流行科学计算库的近乎无缝的替代品。

您可以使用 cuDF 利用 GPU 的计算能力来操作大型数据集。它提供了与 pandas 类似的熟悉界面，但可以更快地处理大得多的数据集。

数据帧 (DataFrames) 基础
cuDF 主要作用于 DataFrame 数据结构。DataFrame 是一种二维数据结构，可以在列中存储不同类型的数据（包括字符、整数、浮点值、分类数据等）。它类似于电子表格、SQL 表或 R 语言中的 data.frame。

cuDF 基础
cuDF 的设计初衷是易于使用。Python 科学计算开发人员会发现 cuDF 在许多方面与 pandas 具有可比性，但重要的是要记住它们之间也存在一些关键差异。

性能：

cuDF：利用 GPU 的并行处理能力，使其在处理大型数据集（GB 到 TB 级）以及连接（joins）、聚合（aggregations）和排序等计算密集型操作时显著加快。
Pandas：在 CPU 上运行，限制了其在处理大型数据集和复杂操作时的性能。
硬件要求：

cuDF：需要 NVIDIA GPU 和 RAPIDS 软件套件。
Pandas：可在任何带 CPU 的系统上运行。
功能：

cuDF：支持大部分 pandas 功能，包括 Series 和 DataFrames 等数据结构，以及常见的操作。然而，某些特性可能略有不同，部分 pandas 函数可能尚未实现或行为有所不同。
Pandas：提供更广泛的函数和特性，包括高级索引和时间序列操作。
兼容性：

cuDF：可以与其它 RAPIDS 库集成，用于 GPU 加速的数据科学工作流。
Pandas：与更广泛的 Python 生态系统（包括 NumPy、Scikit-learn 和 Matplotlib）无缝协作。
Pandas 加速模式： cuDF 提供了一种 cudf.pandas 模式，使用户能够以极小的代码更改来利用 GPU 加速。它充当代理，根据数据大小和操作类型在 cuDF 和 pandas 之间自动切换。

需记住的关键差异：
列名：与 pandas 不同，cuDF 不支持重复的列名。
数据类型：虽然 cuDF 支持大部分 pandas 数据类型，但在处理某些类型（如字符串和分类数据）时可能存在差异。
索引：与 pandas 相比，cuDF 处理索引和多级索引（multi-index）操作的方式可能有所不同。
选择合适的库：
对于小型数据集或受 CPU 限制的任务：由于其更广泛的功能和兼容性，pandas 是一个不错的选择。

对于大型数据集和 GPU 加速计算：cuDF 提供了显著的性能提升，特别是对于数据密集型操作。

cuDF vs. cudf.pandas
您可能会注意到 cuDF 库中有一个可用的 cudf.pandas 模块，这在导入和使用 cuDF 时可能会引起混淆。虽然 cuDF 和 cudf.pandas 都是 RAPIDS 的一部分，旨在通过利用 GPU 的能力来加速数据科学工作流，但仍有一些关键差异需要考虑。最重要的是，用户应该意识到 cuDF 主要在 GPU 上执行，而 cudf.pandas 有时可能会回退（fallback）到 CPU 上的 pandas。

cuDF:

核心库：它是一个 GPU DataFrame 库，提供了针对 GPU 执行优化的 pandas API 子集。
直接访问：当您需要完全控制 GPU 特定功能和操作时，请直接使用它。
性能：由于直接的 GPU 优化，可以为支持的操作提供卓越的性能。
API 兼容性：并非与 pandas 100% 兼容，因此某些 pandas 函数可能不可用或行为不同。
cudf.pandas:

Pandas 加速器：cuDF 之上的一个层，可为您现有的 pandas 代码启用 GPU 加速。
无缝转换：使用它可以无需大幅修改即可加速您的 pandas 代码。
自动回退：如果某个特定操作不受 cuDF 支持，它会自动回退到基于 CPU 的 pandas 实现。
API 兼容性：旨在与 pandas API 实现 100% 兼容，为大多数工作流提供无缝替代。
何时使用：

cuDF：如果您需要最高性能，并习惯于使用略有不同的 API，或者需要利用 GPU 特有的功能。
cudf.pandas：如果您希望以最小的更改加速现有的 pandas 代码，并依赖完整的 pandas API。
cuDF 集成的最新进展：Polars GPU 引擎
Polars 是针对数据科学家和工程师增长最快的 Python 库之一，其设计之初就是为了应对这些挑战。它使用高级查询优化来减少不必要的数据移动和处理，允许数据科学家在单台机器上平滑处理规模达数亿行的工作负载。Polars 弥补了单线程方案太慢而分布式系统又增加了不必要复杂度之间的空白，提供了一个极具吸引力的“中等规模”数据处理解决方案。

cuDF 为 Polars Lazy API 的 Python 用户提供了一个内存中、GPU 加速的执行引擎。该引擎支持大部分核心表达式和数据类型，以及日益增多的高级数据帧操作和数据文件格式。

当使用 GPU 引擎时，Polars 会将表达式转换为优化的查询计划，并确定该计划是否支持在 GPU 上运行。如果不支持，执行过程将透明地回退到标准的 Polars 引擎并在 CPU 上运行。

实用参考链接
cuDF 文档: https://docs.rapids.ai/api/cudf/stable/
cuDF 用户指南: https://docs.rapids.ai/api/cudf/stable/user_guide/
Pandas 文档: https://pandas.pydata.org/docs/
Pandas API 参考: https://pandas.pydata.org/docs/reference/index.html
cuDF 与 Pandas 的差异: https://docs.rapids.ai/api/cudf/stable/user_guide/pandas-comparison/
使用 cuDF 进行数据探索: https://developer.nvidia.com/blog/accelerated-data-analytics-speed-up-data-exploration-with-rapids-cudf/
由 RAPIDS cuDF 驱动的 Polars GPU 引擎现已开启公开测试: https://developer.nvidia.com/blog/polars-gpu-engine-powered-by-rapids-cudf-now-available-in-open-beta/
NVIDIA CUDA-X 现已加速 Polars 数据处理库: https://developer.nvidia.com/blog/nvidia-cuda-x-now-accelerates-the-polars-data-processing-library/
Polars 文档: https://docs.pola.rs/
编码指南
安装
请参考 cuDF RAPIDS 安装指南，了解适用于您的硬件和 Python 环境的安装说明：https://docs.rapids.ai/install/

为了演示示例，我们在下面使用 pip：

python
!pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.8.*
示例：
创建 cuDF 数据帧
python
import cudf
import numpy as np
numRows = 1000000
# 使用 cuDF 创建一个 DataFrame
data = {
    'A': np.random.rand(numRows),
    'B': np.random.rand(numRows),
    'C': np.random.rand(numRows)
}
gdf = cudf.DataFrame(data)
# 显示前几行
print(gdf.head())
探索数据帧
形状 (Shape):

python
gdf.shape
如您所见，第一个值对应于行数，而第二个值表示我们创建的列数。

使用 .info 方法获取数据帧更全面的视角！

python
gdf.info
过滤数据
过滤所有列 'A' 大于 0.5 的行：

python
filtered_gdf = gdf[gdf['A'] > 0.5]
filtered_gdf.shape
print(f"正如您从新过滤的数据帧形状中所见，行数从 {numRows} 减少到了 {filtered_gdf.shape[0]}。这意味着我们过滤掉了 {numRows - filtered_gdf.shape[0]} 行 'A' 值小于 0.5 的数据！")
分组与聚合
创建一个带有类别的全数据帧：

python
data = {
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Value': [10, 20, 30, 40, 50]
}
gdf = cudf.DataFrame(data)
gdf
按类别分组并计算平均值：

python
grouped = gdf.groupby('Category')['Value'].mean().reset_index()
print(grouped)
使用 cuDF vs. cudf.pandas
python
import pandas as pd
import cudf.pandas as xpd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# 直接使用 cuDF - 将数据帧移动到 GPU
gdf = cudf.DataFrame.from_pandas(df)
result = gdf.sum()
# 使用 cudf.pandas
xpd_df = xpd.DataFrame(df)
result = xpd_df.sum()