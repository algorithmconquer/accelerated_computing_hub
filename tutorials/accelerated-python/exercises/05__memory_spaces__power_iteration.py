
import numpy as np
import cupy as cp
import time
from dataclasses import dataclass
from cupyx.profiler import benchmark

# Configuration for the algorithm
@dataclass
class PowerIterationConfig:
    dim: int = 4096                    # Matrix size (dim x dim)
    dominance: float = 0.1             # How much larger the top eigenvalue is (controls convergence speed)
    max_steps: int = 400               # Maximum iterations
    check_frequency: int = 10          # Check for convergence every N steps
    progress: bool = True              # Print progress logs
    residual_threshold: float = 1e-10  # Stop if error is below this


def generate_host(cfg=PowerIterationConfig()):
    """Generates a random diagonalizable matrix on the CPU."""
    np.random.seed(42)

    # Create eigenvalues: One large one (1.0), the rest smaller
    weak_lam = np.random.random(cfg.dim - 1) * (1.0 - cfg.dominance)
    lam = np.random.permutation(np.concatenate(([1.0], weak_lam)))

    # Construct matrix A = P * D * P^-1
    P = np.random.random((cfg.dim, cfg.dim))  # Random invertible matrix
    D = np.diag(np.random.permutation(lam))   # Diagonal matrix of eigenvalues
    A = ((P @ D) @ np.linalg.inv(P))          # The final matrix
    return A


def generate_device_exercise(cfg=PowerIterationConfig()):
    """
    TODO: Generate a random diagonalizable matrix directly on the GPU.
    
    This should mirror the generate_host function but use CuPy instead of NumPy.
    The key benefit: no Host->Device transfer needed!
    """
    # ---------------------------------------------------------
    # TODO 1: Set the random seed on the GPU
    # ---------------------------------------------------------
    cp.random.seed(42)  
    
    # ---------------------------------------------------------
    # TODO 2: Create eigenvalues on the GPU
    # Generate (dim-1) random values, scale them, then combine with 1.0
    # ---------------------------------------------------------
    # TODO: Generate weak eigenvalues using cp.random.random()
    weak_lam = cp.random.random(cfg.dim - 1) * (1.0 - cfg.dominance)
    
    # TODO: Concatenate [1.0] with weak_lam using cp.concatenate and cp.array
    # Then permute them using cp.random.permutation
    lam = cp.random.permutation(cp.concatenate((cp.array([1.0]), weak_lam)))
    
    # ---------------------------------------------------------
    # TODO 3: Construct the matrix A = P * D * P^-1 on the GPU
    # ---------------------------------------------------------
    # TODO: Generate random matrix P using cp.random.random()
    P = cp.random.random((cfg.dim, cfg.dim))
    
    # TODO: Create diagonal matrix D using cp.diag()
    D = cp.diag(cp.random.permutation(lam))
    
    # TODO: Compute A = P @ D @ P^-1 using cp.linalg.inv()
    A = ((P @ D) @ cp.linalg.inv(P)) 
    
    return A


def estimate_host(A, cfg=PowerIterationConfig()):
    """
    Performs power iteration using purely NumPy (CPU).
    """
    # Initialize vector of ones on Host
    x = np.ones(A.shape[0], dtype=np.float64)

    for i in range(0, cfg.max_steps, cfg.check_frequency):
        # Matrix-Vector multiplication
        y = A @ x
        
        # Rayleigh quotient: (x . y) / (x . x)
        lam = (x @ y) / (x @ x)
        
        # Calculate residual (error)
        res = np.linalg.norm(y - lam * x)
        
        # Normalize vector for next step
        x = y / np.linalg.norm(y)

        if cfg.progress:
            print(f"Step {i}: residual = {res:.3e}")

        # Convergence check
        if res < cfg.residual_threshold:
            break

        # Run intermediate steps without checking residual to save compute
        for _ in range(cfg.check_frequency - 1):
            y = A @ x
            x = y / np.linalg.norm(y)

    return (x.T @ (A @ x)) / (x.T @ x)

def estimate_device_exercise(A, cfg=PowerIterationConfig()):
    """
    TODO: Port the power iteration algorithm to the GPU using CuPy.
    
    Steps to complete:
    1. Transfer the input matrix A to the GPU (if it's a numpy array)
    2. Initialize the vector x on the GPU
    3. Replace np operations with cp operations
    4. Return the result as a Python scalar
    """
    # ---------------------------------------------------------
    # TODO 1: MEMORY TRANSFER (Host -> Device)
    # Check if A is a numpy array. If so, move it to GPU using cp.asarray()
    # Otherwise, assume it's already on the device.
    # ---------------------------------------------------------
    if isinstance(A, np.ndarray):
        A_gpu = cp.asarray(A)  # TODO: Transfer to GPU
    else:
        A_gpu = A
    
    # ---------------------------------------------------------
    # TODO 2: Initialize vector of ones ON THE GPU
    # Hint: Use cp.ones() instead of np.ones()
    # ---------------------------------------------------------
    x = cp.ones(A.shape[0], dtype=cp.float64)  # TODO: Create vector of ones on GPU
    
    for i in range(0, cfg.max_steps, cfg.check_frequency):
        # ---------------------------------------------------------
        # TODO 3: Perform GPU computations
        # Replace the operations below with CuPy equivalents
        # ---------------------------------------------------------
        
        # Matrix-Vector multiplication (this works the same with CuPy!)
        y = A_gpu @ x
        
        # Rayleigh quotient
        lam = (x @ y) / (x @ x)
        
        # TODO: Calculate residual using cp.linalg.norm (not np.linalg.norm)
        res = cp.linalg.norm(y - lam * x)
        
        # TODO: Normalize x using cp.linalg.norm
        x = y / cp.linalg.norm(y)
        
        if cfg.progress:
            print(f"Step {i}: residual = {res:.3e}")
        
        if res < cfg.residual_threshold:
            break
        
        for _ in range(cfg.check_frequency - 1):
            y = A_gpu @ x
            x = y / cp.linalg.norm(y)
    
    # ---------------------------------------------------------
    # TODO 4: MEMORY TRANSFER (Device -> Host)
    # Return the eigenvalue as a Python scalar using .item()
    # ---------------------------------------------------------
    result = (x.T @ (A_gpu @ x)) / (x.T @ x)
    return result.item()




# Generate the data on Host
print("Generating Host Data...")
A_host = generate_host()
print(f"Host Matrix Shape: {A_host.shape}")
print(f"Data Type: {A_host.dtype}")

# Run CPU Baseline
print("\nRunning CPU Estimate...")
start_time = time.time()
lam_est_host = estimate_host(A_host)
end_time = time.time()
print(f"\nEstimated Eigenvalue (CPU): {lam_est_host}")
print(f"Time taken: {end_time - start_time:.4f}s")


# Run the GPU implementation
print("\nRunning GPU Estimate (Input is Host Array)...")
start_time = time.time()
lam_est_device = estimate_device_exercise(A_host)
cp.cuda.Stream.null.synchronize()
end_time = time.time()
print(f"\nEstimated Eigenvalue (GPU): {lam_est_device}")
print(f"Time taken: {end_time - start_time:.4f}s")

# Uncomment to test your implementation:
print("\nGenerating Data directly on GPU...")
start_time = time.time()
A_device = generate_device_exercise()
end_time = time.time()
print(f"Generation time: {end_time - start_time:.4f}s")

print("Running GPU Estimate (Input is Device Array)...")
start_time = time.time()
# No transfer overhead here because A_device is already on GPU
lam_est_device_gen = estimate_device_exercise(A_device)
cp.cuda.Stream.null.synchronize()
end_time = time.time()
print(f"Compute time: {end_time - start_time:.4f}s")

print("NumPy:", A_host[0, :3])
print("CuPy:", A_device[0, :3].get())

print("Calculating Reference Eigenvalue (numpy.linalg)...")
# Note: calculating all eigenvalues is computationally expensive
lam_ref = np.linalg.eigvals(A_host).real.max()

print(f"\n--- Results ---")
print(f"Reference: {lam_ref}")
print(f"CPU Est:   {lam_est_host}")
print(f"GPU Est:   {lam_est_device_gen}")

# Assert correctness
np.testing.assert_allclose(lam_est_host, lam_ref, rtol=1e-4)
np.testing.assert_allclose(lam_est_device_gen, lam_ref, rtol=1e-4)
print("\nAccuracy verification passed!")


cfg = PowerIterationConfig(progress=False)

# 1. CPU
print("Timing CPU...")
result_cpu = benchmark(estimate_host, args=(A_host, cfg), n_repeat=10)
t_cpu_ms = result_cpu.cpu_times.mean() * 1000

# 2. GPU (with transfer overhead)
print("Timing GPU (Host Input)...")
result_transfer = benchmark(estimate_device_exercise, args=(A_host, cfg), n_repeat=10)
t_gpu_transfer_ms = result_transfer.gpu_times.mean() * 1000

# 3. GPU (pure device)
print("Timing GPU (Device Input)...")
result_pure = benchmark(estimate_device_exercise, args=(A_device, cfg), n_repeat=10)
t_gpu_pure_ms = result_pure.gpu_times.mean() * 1000

print(f"\n--- Average Compute Times ---")
print(f"CPU:                 {t_cpu_ms:.2f} ms")
print(f"GPU (with transfer): {t_gpu_transfer_ms:.2f} ms")
print(f"GPU (pure):          {t_gpu_pure_ms:.2f} ms")

speedup = t_cpu_ms / t_gpu_pure_ms
print(f"\nSpeedup: {speedup:.1f}x")



'''
--- Average Compute Times ---
CPU:                 278.74 ms
GPU (with transfer): 53.99 ms
GPU (pure):          32.88 ms

Speedup: 8.5x
'''



