from cuda.core.experimental import system, Device
import numpy as np

def func1():
    # Check CUDA driver version
    print(f"CUDA driver version: {system.driver_version}")
    
    # Get number of available devices
    print(f"Number of CUDA devices: {system.num_devices}")
    
    # Get device information
    if system.num_devices > 0:
        device = Device(0)  # Get the first GPU
        device.set_current()  # Tell CUDA we want to use this GPU
        print(f"Device name: {device.name}")
        print(f"Device UUID: {device.uuid}")
        print(f"PCI Bus ID: {device.pci_bus_id}")
    else:
        print("No CUDA devices found!")



def func2():
    # Get the first GPU device (device 0)
    device = Device(0)
    
    # Set as current device (this creates and activates a context)
    device.set_current()
    
    print(f"Using device: {device.name}")
    print(f"Device id: {device.device_id}")



def func3():
    device = Device(0)
    device.set_current()
    
    # Get device properties
    props = device.properties
    print(f"Device name: {device.name}")
    print(f"Compute capability: {props.compute_capability_major}.{props.compute_capability_minor}")
    print(f"Multiprocessor count: {props.multiprocessor_count}")
    print(f"Max threads per block: {props.max_threads_per_block}")
    print(f"Max block dimensions: ({props.max_block_dim_x}, {props.max_block_dim_y}, {props.max_block_dim_z})")


def func4():
    import numpy as np
    from cuda.core.experimental import Device
    
    # Initialize our GPU
    device = Device(0)
    device.set_current()
    
    # Calculate how much memory we need
    # We want to store 1000 float32 numbers
    # Each float32 takes 4 bytes, so we need 1000 * 4 = 4000 bytes
    size_bytes = 1000 * 4
    
    # Allocate memory on the GPU
    device_buffer = device.allocate(size_bytes)
    
    print(f"Allocated {size_bytes} bytes on GPU")
    print(f"Buffer memory address: {device_buffer}")


def func5():
    from cuda.core.experimental import Device, Program
    import numpy as np
    
    # CUDA C++ source code for our kernel
    vector_add_source = """
    extern "C" __global__ void vector_add(float *a, float *b, float *c, int n) {
        // Each thread calculates its unique index
        int i = threadIdx.x + blockIdx.x * blockDim.x;
    
        // Make sure we don't go beyond our array bounds
        if (i < n) {
            c[i] = a[i] + b[i];  // Add corresponding elements
        }
    }
    """
    
    # Initialize our GPU
    device = Device(0)
    device.set_current()
    
    # Compile the CUDA code into a program
    program = Program(vector_add_source, code_type='c++')
    compiled_program = program.compile(target_type='cubin')
    
    # Get the specific kernel function we want to use
    kernel = compiled_program.get_kernel("vector_add")
    
    print("Kernel compiled successfully!")


def func6():
    import cupy as cp
    from cuda.core.experimental import launch, LaunchConfig, ProgramOptions
    from cuda.core.experimental import Device, Program
    import numpy as np
    
    # CUDA C++ source code for our kernel
    vector_add_source = """
    extern "C" __global__ void vector_add(float *a, float *b, float *c, int n) {
        // Each thread calculates its unique index
        int i = threadIdx.x + blockIdx.x * blockDim.x;
    
        // Make sure we don't go beyond our array bounds
        if (i < n) {
            c[i] = a[i] + b[i];  // Add corresponding elements
        }
    }
    """
    
    # Initialize our GPU
    device = Device(0)
    device.set_current()
    
    # Compile the CUDA code into a program
    program = Program(vector_add_source, code_type='c++')
    compiled_program = program.compile(target_type='cubin')
    
    # Get the specific kernel function we want to use
    kernel = compiled_program.get_kernel("vector_add")
    
    def execute_vector_add():
        # Initialize device and create a stream
        device = Device(0)
        device.set_current()
        s = device.create_stream()
    
        # Prepare our test data
        N = 1000  # Number of elements
        a = np.arange(N, dtype=np.float32)      # [0, 1, 2, ..., 999]
        b = np.arange(N, dtype=np.float32)      # [0, 1, 2, ..., 999]
        print(f"Input arrays have {N} elements each")
    
        # Step 2: Copy input data from CPU to GPU
        d_a = cp.asarray(a)
        d_b = cp.asarray(b)
        d_c = cp.empty(N, dtype=cp.float32)
    
        # Configure how to launch the kernel
        block_size = 256  # Number of threads per block
        grid_size = (N + block_size - 1) // block_size  # Number of blocks needed
    
        print(f"Launch configuration: {grid_size} blocks of {block_size} threads each")
        print(f"Total threads: {grid_size * block_size}")
    
        # Create the launch configuration
        config = LaunchConfig(grid=(grid_size,), block=(block_size,))
    
        # Launch the kernel
        launch(s, config, kernel, d_a.data.ptr, d_b.data.ptr, d_c.data.ptr, cp.uint64(N))
        s.sync() # Wait for kernel to complete
        print("Kernel launched and executed")
    
        # Copy the result back from GPU to CPU
        c = cp.asnumpy(d_c)
        print("Results copied back to CPU")
    
        return c
    
    # Execute our vector addition
    result = execute_vector_add()
    
    # Verify the result
    expected = np.arange(1000, dtype=np.float32) * 2  # [0, 2, 4, ..., 1998]
    success = np.allclose(result, expected)
    print(f"Kernel execution successful: {success}")
    print(f"First 10 results: {result[:10]}")
    print(f"Expected first 10: {expected[:10]}")


def func7():
    import cupy as cp
    from cuda.core.experimental import launch, LaunchConfig, ProgramOptions, Program
    # Templated matrix multiplication kernel
    matmul_source = """
    template<typename T>
    __global__ void matrix_multiply(const T *A, const T *B, T *C, size_t N) {
        // Calculate which row and column this thread handles
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
    
        // Make sure we're within the matrix bounds
        if (row < N && col < N) {
            T sum = T(0);
    
            // Compute dot product of row from A and column from B
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
    
            // Store the result
            C[row * N + col] = sum;
        }
    }
    """
    
    def matrix_multiply_gpu(A, B):
        # Initialize device and create a stream
        device = Device(0)
        device.set_current()
        s = device.create_stream()
    
        N = A.shape[0]
        assert A.shape == (N, N) and B.shape == (N, N), "Matrices must be square and same size"
        print(f"Multiplying {N}x{N} matrices")
    
        # Compile the templated matrix multiplication kernel with specific C++ compiler flags
        program_options = ProgramOptions(std="c++17")
        program = Program(matmul_source, code_type='c++', options=program_options)
        compiled_program = program.compile(target_type='cubin', name_expressions=("matrix_multiply<float>",))
        kernel = compiled_program.get_kernel("matrix_multiply<float>")
    
        # Copy input matrices to GPU
        d_A = cp.asarray(A)
        d_B = cp.asarray(B)
        d_C = cp.empty((N, N), dtype=cp.float32)
    
        # Configure 2D launch (threads arranged in a 2D grid)
        block_size = 16  # 16x16 = 256 threads per block
        grid_size = (N + block_size - 1) // block_size
    
        print(f"Launch config: {grid_size}x{grid_size} blocks of {block_size}x{block_size} threads")
    
        # Create 2D launch configuration
        config = LaunchConfig(grid=(grid_size, grid_size), block=(block_size, block_size))
    
        # Launch the kernel
        launch(s, config, kernel, d_A.data.ptr, d_B.data.ptr, d_C.data.ptr, cp.uint64(N))
        s.sync()
    
        # Copy result back
        C = cp.asnumpy(d_C)
    
        print("Matrix multiplication completed on GPU")
        return C
    
    print("Testing matrix multiplication...")
    A = np.random.random((64, 64)).astype(np.float32)
    B = np.random.random((64, 64)).astype(np.float32)
    
    # Compare GPU result with CPU result
    C_gpu = matrix_multiply_gpu(A, B)
    C_cpu = np.dot(A, B) # NumPy's optimized matrix multiplication
    
    # Check if results match (within floating-point precision)
    matches = np.allclose(C_gpu, C_cpu, atol=1e-5)
    print(f"GPU and CPU results match: {matches}")
    
    if matches:
        print("Success! Matrix multiplication kernel works correctly.")
    else:
        print("Results don't match - there might be a bug in the kernel.")


def func8():
    # Compile your kernel here
    from cuda.core.experimental import Device, Program
    
    multiply_kernel_source = """
    // TODO: Implement vector multiplication kernel
    extern "C" __global__ void vector_multiply(float *a, float *b, float *c, int n) {
        // Each thread calculates its unique index
        int i = threadIdx.x + blockIdx.x * blockDim.x;
    
        // Make sure we don't go beyond our array bounds
        if (i < n) {
            c[i] = a[i] * b[i];  // Multiply corresponding elements
        }
    }
    """
    
    # Initialize our GPU
    device = Device(0)
    device.set_current()
    
    # Compile the CUDA code into a program
    program = Program(multiply_kernel_source, code_type='c++')
    compiled_program = program.compile(target_type='cubin')
    
    # Get the specific kernel function we want to use
    kernel = compiled_program.get_kernel("vector_multiply")
    
    print("Kernel compiled successfully!")

    # Launch your kernel here
    import numpy as np
    import cupy as cp
    from cuda.core.experimental import Device, launch, LaunchConfig
    
    def vector_multiply(a, b):
        # Step 1: Initialize device
        device = Device(0)
        device.set_current()
        s = device.create_stream()
    
        # Step 2: Prepare our test data
        N = 1000  # Number of elements
        a = np.arange(N, dtype=np.float32)      # [0, 1, 2, ..., 999]
        b = np.arange(N, dtype=np.float32)      # [0, 1, 2, ..., 999]
        print(f"Input arrays have {N} elements each")
    
        # Step 3: Copy input data from CPU to GPU
        d_a = cp.asarray(a)
        d_b = cp.asarray(b)
        d_c = cp.ones(N, dtype=cp.float32)
    
        # Step 4: Configure how to launch the kernel
        block_size = 256  # Number of threads per block
        grid_size = (N + block_size - 1) // block_size  # Number of blocks needed
    
        print(f"Launch configuration: {grid_size} blocks of {block_size} threads each")
        print(f"Total threads: {grid_size * block_size}")
    
        # Create the launch configuration
        config = LaunchConfig(grid=(grid_size,), block=(block_size,))
        ker_args = (d_a.data.ptr, d_b.data.ptr, d_c.data.ptr, N)
    
        # Step 5: Launch the kernel!
        launch(s, config, kernel, *ker_args)
        s.sync()
        print("Kernel launched and executed")
    
        # Step 6: Copy the result back from GPU to CPU
        c = cp.asnumpy(d_c)
        print("Results copied back to CPU")
    
        return c
    
    # Execute our vector addition
    a = np.arange(1000, dtype=np.float32)
    result = vector_multiply(a, a)
    
    # Verify the result
    expected = a * a
    success = np.allclose(result, expected)
    print(f"Kernel execution successful: {success}")
    print(f"First 10 results: {result[:10]}")
    print(f"Expected first 10: {expected[:10]}")


func6()













