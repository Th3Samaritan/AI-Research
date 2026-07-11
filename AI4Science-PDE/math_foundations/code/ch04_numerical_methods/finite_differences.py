"""
Chapter 4: Numerical Methods - Finite Differences
FD stencil construction and convergence study.
"""
import numpy as np

def derivative_fd(f, x, dx, order=2, type='central'):
    """
    Compute finite difference derivative.
    order: accuracy order (1, 2)
    type: 'forward', 'backward', 'central'
    """
    if type == 'forward':
        if order == 1:
            return (f(x + dx) - f(x)) / dx
        elif order == 2:
            return (-f(x + 2*dx) + 4*f(x + dx) - 3*f(x)) / (2*dx)
    elif type == 'backward':
        if order == 1:
            return (f(x) - f(x - dx)) / dx
        elif order == 2:
            return (3*f(x) - 4*f(x - dx) + f(x - 2*dx)) / (2*dx)
    elif type == 'central':
        if order == 2:
            return (f(x + dx) - f(x - dx)) / (2*dx)
        elif order == 4:
            return (-f(x + 2*dx) + 8*f(x + dx) - 8*f(x - dx) + f(x - 2*dx)) / (12*dx)
            
    raise ValueError("Unsupported FD type or order")

if __name__ == "__main__":
    print("--- FD Convergence Study ---")
    # Test function f(x) = sin(x), f'(x) = cos(x)
    f = np.sin
    x0 = 1.0
    exact = np.cos(x0)
    
    dx_vals = [1e-1, 1e-2, 1e-3, 1e-4]
    
    print(f"Exact derivative at x={x0}: {exact:.6f}")
    print(f"{'dx':<10} | {'Forward(1)':<15} | {'Central(2)':<15} | {'Central(4)':<15}")
    print("-" * 60)
    
    for dx in dx_vals:
        df_f1 = derivative_fd(f, x0, dx, order=1, type='forward')
        df_c2 = derivative_fd(f, x0, dx, order=2, type='central')
        df_c4 = derivative_fd(f, x0, dx, order=4, type='central')
        
        err_f1 = abs(df_f1 - exact)
        err_c2 = abs(df_c2 - exact)
        err_c4 = abs(df_c4 - exact)
        
        print(f"{dx:<10.0e} | {err_f1:<15.2e} | {err_c2:<15.2e} | {err_c4:<15.2e}")
