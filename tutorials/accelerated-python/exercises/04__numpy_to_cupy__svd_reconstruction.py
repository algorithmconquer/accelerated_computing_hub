


def func111():
    import matplotlib.pyplot as plt
    import cv2
    import numpy as xp
    import cupy as cp

    image = cv2.imread("loonie_grass.jpg", cv2.IMREAD_GRAYSCALE)
    
    print(f"nbytes: {image.nbytes}")
    print(f"shape: {image.shape}")
    print(image)
    '''
    plt.imshow(image, cmap="gray")
    plt.title("Bryce's Dog")
    plt.show()
    '''
    U, S, Vt = xp.linalg.svd(image, full_matrices=False)
    print(U.shape, S.shape, Vt.shape)
    print(S[:10])
    #plt.semilogy(S)
    #plt.savefig('semilogy.png')
    #plt.show()
    
    # First 3 terms.
    '''
    nterms = 3
    reconstructed = U[:, :nterms] @ xp.diag(S[:nterms]) @ Vt[:nterms, :]
    plt.imshow(reconstructed, cmap="gray")
    plt.title(f"n = {nterms}")
    plt.savefig(f'svd_top{nterms}.png')
    plt.show()
    '''
    
    plt.figure(figsize=(16, 4))
    
    start, end, step = 10, 50, 10
    for i in range(start, end, step):
      plt.subplot(1, (end - start) // step + 1, (i - start) // step + 1)
      reconstructed = U[:, :i] @ xp.diag(S[:i]) @ Vt[:i, :]
      compress_ratio = (U[:, :i].nbytes + S[:i].nbytes + Vt[:i, :].nbytes) / image.nbytes
      print(f"n = {i}: compression = {compress_ratio:.1%}")
      plt.imshow(reconstructed, cmap="gray")
      plt.title(f"n = {i}")
    
    plt.tight_layout()
    plt.savefig(f'svd_10_50.png')
    plt.show()



def func222():
    import cv2
    import numpy as np
    import cupy as cp
    import cupyx as cpx # For `cupyx.profiler.benchmark`.
    cpu_image = cv2.imread('loonie_grass.jpg', cv2.IMREAD_GRAYSCALE)
    gpu_image = cp.asarray(cpu_image)
    repeat = 10
    warmup = 1
    D_np = cpx.profiler.benchmark(n_repeat=repeat, n_warmup=warmup, func=lambda: np.linalg.svd(cpu_image, full_matrices=False)).cpu_times
    D_cp = cpx.profiler.benchmark(n_repeat=repeat, n_warmup=warmup, func=lambda: cp.linalg.svd(gpu_image, full_matrices=False)).cpu_times

    print(f"SVD (Host)   = {D_np.mean():.3g} s ± {(D_np.std() / D_np.mean()):.2%} (mean ± relative stdev of {D_np.size} runs)")
    print(f"SVD (Device) = {D_cp.mean():.3g} s ± {(D_cp.std() / D_cp.mean()):.2%} (mean ± relative stdev of {D_cp.size} runs)")

def func333():
    import cv2
    import numpy as np
    import cupy as cp
    import cupyx as cpx # For `cupyx.profiler.benchmark`.
    cpu_image = cv2.imread('loonie_grass.jpg', cv2.IMREAD_GRAYSCALE)
    cpu_image_tile = np.tile(cpu_image, (2, 2))
    gpu_image_tile = cp.asarray(cpu_image_tile)
    repeat = 10
    warmup = 1
    D_np = cpx.profiler.benchmark(n_repeat=repeat, n_warmup=warmup, func=lambda:np.linalg.svd(cpu_image_tile, full_matrices=False)).cpu_times
    D_cp = cpx.profiler.benchmark(n_repeat=repeat, n_warmup=warmup, func=lambda:cp.linalg.svd(gpu_image_tile, full_matrices=False)).cpu_times
    
    print(f"SVD (Host)   = {D_np.mean():.3g} s ± {(D_np.std() / D_np.mean()):.2%} (mean ± relative stdev of {D_np.size} runs)")
    print(f"SVD (Device) = {D_cp.mean():.3g} s ± {(D_cp.std() / D_cp.mean()):.2%} (mean ± relative stdev of {D_cp.size} runs)")


func222()
"""
SVD (Host)   = 4.09 s ± 8.21% (mean ± relative stdev of 10 runs)
SVD (Device) = 0.423 s ± 0.15% (mean ± relative stdev of 10 runs)
"""
func333()
"""
SVD (Host)   = 17.6 s ± 2.11% (mean ± relative stdev of 10 runs)
SVD (Device) = 1.13 s ± 0.61% (mean ± relative stdev of 10 runs)
"""














