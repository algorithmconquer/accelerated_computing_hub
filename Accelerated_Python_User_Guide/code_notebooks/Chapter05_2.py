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
    i = cuda.grid(1)
    if i < input_array.size:
        output_array[i] = input_array[i]

foo[1, len(input)](input, output)
logging.info(f"------output={output}")