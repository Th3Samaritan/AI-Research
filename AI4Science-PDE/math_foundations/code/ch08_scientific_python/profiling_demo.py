"""
Chapter 8: Profiling and Identifying Bottlenecks
Note: To use line_profiler, run: kernprof -l -v profiling_demo.py
If line_profiler is not installed, the script will mock it.
"""
import numpy as np
import time

try:
    from line_profiler import profile
except ImportError:
    # Dummy decorator if line_profiler is not installed
    def profile(func):
        return func

@profile
def slow_pde_step(u, dx, dt, alpha):
    """Naive, slow implementation of 1D heat equation step using loops."""
    nx = len(u)
    un = u.copy()
    for i in range(1, nx - 1):
        un[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    return un

@profile
def fast_pde_step(u, dx, dt, alpha):
    """Fast, vectorized implementation of 1D heat equation step."""
    un = u.copy()
    un[1:-1] = u[1:-1] + alpha * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    return un

def main():
    nx = 10000
    dx = 0.01
    dt = 0.0001
    alpha = 0.1
    
    u_init = np.sin(np.pi * np.linspace(0, 1, nx))
    
    print("Running naive implementation (100 steps)...")
    t0 = time.time()
    u = u_init.copy()
    for _ in range(100):
        u = slow_pde_step(u, dx, dt, alpha)
    t1 = time.time()
    print(f"Naive time: {t1 - t0:.4f} seconds")
    
    print("Running vectorized implementation (100 steps)...")
    t0 = time.time()
    u = u_init.copy()
    for _ in range(100):
        u = fast_pde_step(u, dx, dt, alpha)
    t1 = time.time()
    print(f"Vectorized time: {t1 - t0:.4f} seconds")
    
    print("\nTo see line-by-line profile, install line_profiler and run:")
    print("kernprof -l -v profiling_demo.py")

if __name__ == "__main__":
    main()
