"""
Chapter 3: PDEs - Poisson Solver
2D Poisson equation with Jacobi, Gauss-Seidel, and SOR iterative methods.
"""
import numpy as np

def poisson_2d_jacobi(f, dx, dy, max_iter=1000, tol=1e-5):
    """Solve Laplace/Poisson eqn using Jacobi iteration."""
    ny, nx = f.shape
    u = np.zeros_like(f)
    u_new = np.zeros_like(u)
    
    dx2, dy2 = dx**2, dy**2
    factor = 0.5 / (dx2 + dy2)
    
    for it in range(max_iter):
        u_new[1:-1, 1:-1] = factor * (
            dy2 * (u[1:-1, 2:] + u[1:-1, :-2]) + 
            dx2 * (u[2:, 1:-1] + u[:-2, 1:-1]) - 
            dx2 * dy2 * f[1:-1, 1:-1]
        )
        
        # Dirichlet boundaries (keep 0)
        
        if np.max(np.abs(u_new - u)) < tol:
            print(f"Jacobi converged in {it+1} iterations.")
            return u_new
        u = u_new.copy()
        
    print("Jacobi reached max iterations without full convergence.")
    return u

def poisson_2d_gauss_seidel(f, dx, dy, max_iter=1000, tol=1e-5):
    """Solve using Gauss-Seidel iteration (in-place updates)."""
    ny, nx = f.shape
    u = np.zeros_like(f)
    
    dx2, dy2 = dx**2, dy**2
    factor = 0.5 / (dx2 + dy2)
    
    for it in range(max_iter):
        max_diff = 0.0
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                old_val = u[i, j]
                u[i, j] = factor * (
                    dy2 * (u[i, j+1] + u[i, j-1]) + 
                    dx2 * (u[i+1, j] + u[i-1, j]) - 
                    dx2 * dy2 * f[i, j]
                )
                max_diff = max(max_diff, abs(u[i, j] - old_val))
                
        if max_diff < tol:
            print(f"Gauss-Seidel converged in {it+1} iterations.")
            return u
            
    print("Gauss-Seidel reached max iterations.")
    return u

def poisson_2d_sor(f, dx, dy, omega=1.5, max_iter=1000, tol=1e-5):
    """Solve using Successive Over-Relaxation (SOR)."""
    ny, nx = f.shape
    u = np.zeros_like(f)
    
    dx2, dy2 = dx**2, dy**2
    factor = 0.5 / (dx2 + dy2)
    
    for it in range(max_iter):
        max_diff = 0.0
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                old_val = u[i, j]
                new_val = factor * (
                    dy2 * (u[i, j+1] + u[i, j-1]) + 
                    dx2 * (u[i+1, j] + u[i-1, j]) - 
                    dx2 * dy2 * f[i, j]
                )
                u[i, j] = (1 - omega) * old_val + omega * new_val
                max_diff = max(max_diff, abs(u[i, j] - old_val))
                
        if max_diff < tol:
            print(f"SOR (w={omega}) converged in {it+1} iterations.")
            return u
            
    print("SOR reached max iterations.")
    return u

if __name__ == "__main__":
    print("--- 2D Poisson Equation ---")
    nx, ny = 20, 20
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    f = np.zeros((ny, nx))
    f[ny//2, nx//2] = -100.0 # Source term in center
    
    u_jac = poisson_2d_jacobi(f, dx, dy, max_iter=2000)
    u_gs = poisson_2d_gauss_seidel(f, dx, dy, max_iter=2000)
    u_sor = poisson_2d_sor(f, dx, dy, omega=1.7, max_iter=2000)
