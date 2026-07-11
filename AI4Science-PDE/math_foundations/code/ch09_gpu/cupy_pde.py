"""
Chapter 9: Porting 2D Heat Equation to CuPy
NOTE: This script requires an NVIDIA GPU and the 'cupy' library installed.
"""
import numpy as np
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("WARNING: CuPy is not installed. GPU code will not execute. Needs NVIDIA GPU.")

def solve_heat_2d_numpy(nx, ny, nt, alpha, dx, dy, dt):
    """CPU implementation using NumPy."""
    u = np.zeros((nx, ny))
    u[nx//4:3*nx//4, ny//4:3*ny//4] = 100.0 # Hot square in the middle
    
    for n in range(nt):
        u[1:-1, 1:-1] = (u[1:-1, 1:-1] + 
                         alpha * dt / dx**2 * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) +
                         alpha * dt / dy**2 * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]))
    return u

def solve_heat_2d_cupy(nx, ny, nt, alpha, dx, dy, dt):
    """GPU implementation using CuPy."""
    if not HAS_CUPY:
        return None
    
    u = cp.zeros((nx, ny))
    u[nx//4:3*nx//4, ny//4:3*ny//4] = 100.0
    
    # CuPy uses exactly the same syntax as NumPy!
    for n in range(nt):
        u[1:-1, 1:-1] = (u[1:-1, 1:-1] + 
                         alpha * dt / dx**2 * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) +
                         alpha * dt / dy**2 * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]))
    
    # Synchronize to make sure GPU is done before returning (for timing)
    cp.cuda.Stream.null.synchronize()
    return u

def main():
    nx, ny = 500, 500
    nt = 1000
    alpha = 1e-4
    dx, dy = 0.01, 0.01
    dt = (dx**2 * dy**2) / (2 * alpha * (dx**2 + dy**2)) * 0.5 # Stability condition

    print(f"Grid size: {nx}x{ny}, Time steps: {nt}")
    
    # NumPy CPU benchmark
    t0 = time.time()
    u_np = solve_heat_2d_numpy(nx, ny, nt, alpha, dx, dy, dt)
    t1 = time.time()
    t_cpu = t1 - t0
    print(f"NumPy (CPU) time: {t_cpu:.4f} seconds")

    # CuPy GPU benchmark
    if HAS_CUPY:
        # Warm-up (GPU compilation overhead on first run)
        _ = solve_heat_2d_cupy(10, 10, 1, alpha, dx, dy, dt)
        
        t0 = time.time()
        u_cp = solve_heat_2d_cupy(nx, ny, nt, alpha, dx, dy, dt)
        t1 = time.time()
        t_gpu = t1 - t0
        print(f"CuPy (GPU) time: {t_gpu:.4f} seconds")
        
        speedup = t_cpu / t_gpu
        print(f"GPU Speedup: {speedup:.2f}x")
    else:
        print("Skipping GPU benchmark due to missing CuPy.")

if __name__ == "__main__":
    main()
