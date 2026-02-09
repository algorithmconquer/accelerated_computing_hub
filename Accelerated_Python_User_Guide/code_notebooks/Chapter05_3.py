import numpy as np
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s')


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
foo[grid_blocks, block_threads](input, output)
logging.info(f"------output={output}")
