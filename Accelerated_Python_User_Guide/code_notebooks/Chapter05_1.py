import cupy
from numba import cuda
from numba import config as numba_config
numba_config.CUDA_ENABLE_PYNVJITLINK = True
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d]-%(levelname)s: %(message)s')



@cuda.jit
def add(x_array, y_array, output_array):
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(start, x_array.shape[0], stride):
                output_array[i] = x_array[i] + y_array[i]

logging.info(f"------Numba 与 CuPy 配合使用------")
a = cupy.arange(10)
b = a * 2
out = cupy.zeros_like(a)
add[1, 32](a, b, out)
logging.info(f"------out={out}") # => [ 0  3  6  9 12 15 18 21 24 27]
