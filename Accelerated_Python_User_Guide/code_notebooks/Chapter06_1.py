from numba import cuda
from nvmath.device import random
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s')

compiled_apis = random.Compile()
threads, blocks = 128, 128
nthreads = blocks * threads
states = random.StatesPhilox4_32_10(nthreads)
# 接下来，定义并启动一个设置内核，它将使用 nvmath.device.random.init 函数初始化状态。
@cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
def setup(states):
    i = cuda.grid(1)
    random.init(1234, i, 0, states[i])

setup[blocks, threads](states)
# 准备好状态数组后，您可以使用诸如 nvmath.device.random.normal2 之类的采样器
# 在内核中采样随机值。
@cuda.jit(link=compiled_apis.files, extensions=compiled_apis.extension)
def kernel(states):
    i = cuda.grid(1)
    random_values = random.normal2(states[i])





