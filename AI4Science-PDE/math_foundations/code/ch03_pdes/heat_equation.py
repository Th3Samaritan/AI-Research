"""
Chapter 3: PDEs - Heat Equation
1D and 2D heat equation with explicit & implicit schemes, CFL stability.
"""
import numpy as np

def heat_1d_explicit(u0, alpha, dt, dx, n_steps):
    """Solve 1D heat equation using Forward Time Centered Space (FTCS)."""
    r = alpha * dt / (dx**2)
    if r > 0.5:
        print(f"Warning: r = {r} > 0.5. Scheme may be unstable (CFL condition violated).")
        
    nx = len(u0)
    u = u0.copy()
    u_new = np.zeros(nx)
    
    history = [u.copy()]
    
    for _ in range(n_steps):
        # Explicit update (interior points)
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        # Dirichlet boundaries (fixed to 0 for simplicity, or just keep u0)
        u_new[0] = u[0]
        u_new[-1] = u[-1]
        
        u = u_new.copy()
        history.append(u.copy())
        
    return history

def heat_1d_implicit(u0, alpha, dt, dx, n_steps):
    """Solve 1D heat equation using Backward Time Centered Space (BTCS) - Implicit."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    
    r = alpha * dt / (dx**2)
    nx = len(u0)
    
    # Construct tridiagonal matrix for (I - r * D2)
    main_diag = (1 + 2*r) * np.ones(nx - 2)
    off_diag = -r * np.ones(nx - 3)
    diags = [main_diag, off_diag, off_diag]
    A = sp.diags(diags, offsets=[0, -1, 1], format='csr')
    
    u = u0.copy()
    history = [u.copy()]
    
    for _ in range(n_steps):
        # RHS incorporates current state and boundary conditions
        b = u[1:-1].copy()
        b[0] += r * u[0]
        b[-1] += r * u[-1]
        
        # Solve tridiagonal system
        u_interior = spla.spsolve(A, b)
        u[1:-1] = u_interior
        history.append(u.copy())
        
    return history

if __name__ == "__main__":
    print("--- 1D Heat Equation ---")
    nx = 50
    dx = 1.0 / (nx - 1)
    x = np.linspace(0, 1, nx)
    
    # Initial condition: half-sine wave
    u0 = np.sin(np.pi * x)
    
    alpha = 0.01
    dt_explicit = 0.0001
    dt_implicit = 0.01 # Larger dt allowed!
    
    hist_exp = heat_1d_explicit(u0, alpha, dt_explicit, dx, n_steps=100)
    print(f"Explicit FTCS: final max val {np.max(hist_exp[-1]):.4f}")
    
    hist_imp = heat_1d_implicit(u0, alpha, dt_implicit, dx, n_steps=10)
    print(f"Implicit BTCS: final max val {np.max(hist_imp[-1]):.4f}")
