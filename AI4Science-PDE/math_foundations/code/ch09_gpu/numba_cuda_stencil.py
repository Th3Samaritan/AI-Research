"""
Chapter 9: Numba CUDA Kernel for Finite-Difference Stencil
NOTE: This script requires an NVIDIA GPU and the 'numba' library installed.
"""
import numpy as np
import time
import math

try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except ImportError:
    HAS_CUDA = False
    print("WARNING: Numba is not installed or CUDA is not available.")

def solve_cpu(u, alpha, dx, dt, nt):
    nx = u.shape[0]
    un = np.empty_like(u)
    for _ in range(nt):
        un[:] = u
        for i in range(1, nx - 1):
            u[i] = un[i] + alpha * dt / dx**2 * (un[i+1] - 2*un[i] + un[i-1])
    return u

if HAS_CUDA:
    @cuda.jit
    def heat_1d_kernel(u_old, u_new, alpha_dt_dx2, nx):
        """CUDA kernel for 1D heat equation step."""
        i = cuda.grid(1)
        if 0 < i < nx - 1:
            u_new[i] = u_old[i] + alpha_dt_dx2 * (u_old[i+1] - 2*u_old[i] + u_old[i-1])

def solve_gpu(u_init, alpha, dx, dt, nt):
    if not HAS_CUDA:
        return None
        
    nx = u_init.shape[0]
    alpha_dt_dx2 = alpha * dt / dx**2
    
    # Allocate device arrays
    d_u = cuda.to_device(u_init)
    d_un = cuda.device_array_like(u_init)
    
    # Configure kernel grid
    threads_per_block = 256
    blocks_per_grid = math.ceil(nx / threads_per_block)
    
    for n in range(nt):
        # Swap arrays by passing alternately
        if n % 2 == 0:
            heat_1d_kernel[blocks_per_grid, threads_per_block](d_u, d_un, alpha_dt_dx2, nx)
        else:
            heat_1d_kernel[blocks_per_grid, threads_per_block](d_un, d_u, alpha_dt_dx2, nx)
            
    # Retrieve result based on final iteration parity
    if nt % 2 == 1:
        return d_un.copy_to_host()
    else:
        return d_u.copy_to_host()

def main():
    nx = 1000000
    nt = 100
    alpha = 0.1
    dx = 1.0 / nx
    dt = 0.1 * dx**2 / alpha
    
    print(f"1D Grid size: {nx}, Time steps: {nt}")
    u_init = np.zeros(nx, dtype=np.float64)
    u_init[nx//2 - 50:nx//2 + 50] = 100.0
    
    # CPU
    u_cpu = u_init.copy()
    t0 = time.time()
    solve_cpu(u_cpu, alpha, dx, dt, nt)
    t1 = time.time()
    t_cpu = t1 - t0
    print(f"CPU time: {t_cpu:.4f} seconds")
    
    # GPU
    if HAS_CUDA:
        # Warmup
        _ = solve_gpu(np.zeros(100), alpha, dx, dt, 1)
        
        t0 = time.time()
        u_gpu = solve_gpu(u_init, alpha, dx, dt, nt)
        t1 = time.time()
        t_gpu = t1 - t0
        print(f"Numba CUDA time: {t_gpu:.4f} seconds")
        print(f"GPU Speedup: {t_cpu/t_gpu:.2f}x")
        
        # Verify correctness
        diff = np.max(np.abs(u_cpu - u_gpu))
        print(f"Max difference between CPU and GPU: {diff:e}")
    else:
        print("Skipping CUDA benchmark.")

if __name__ == "__main__":
    main()
