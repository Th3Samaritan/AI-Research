"""
Chapter 3: PDEs - Wave Equation
1D wave equation, string vibration, reflection.
"""
import numpy as np

def wave_1d(u0, v0, c, dt, dx, n_steps):
    """
    Solve 1D wave equation using explicit central differences.
    u_tt = c^2 * u_xx
    """
    C = c * dt / dx
    if C > 1.0:
        print(f"Warning: Courant number C = {C} > 1.0. Scheme is unstable.")
        
    nx = len(u0)
    u = u0.copy()
    u_prev = np.zeros_like(u0)
    u_new = np.zeros_like(u0)
    
    # First time step using initial velocity v0
    u_prev = u.copy()
    # Centered difference approximation for initial step
    for i in range(1, nx - 1):
        u[i] = u_prev[i] + dt * v0[i] + 0.5 * C**2 * (u_prev[i+1] - 2*u_prev[i] + u_prev[i-1])
        
    # Dirichlet boundaries
    u[0] = 0; u[-1] = 0
    
    history = [u_prev.copy(), u.copy()]
    
    for _ in range(n_steps - 1):
        for i in range(1, nx - 1):
            u_new[i] = 2*u[i] - u_prev[i] + C**2 * (u[i+1] - 2*u[i] + u[i-1])
            
        u_new[0] = 0
        u_new[-1] = 0
        
        u_prev = u.copy()
        u = u_new.copy()
        history.append(u.copy())
        
    return history

if __name__ == "__main__":
    print("--- 1D Wave Equation ---")
    nx = 100
    L = 1.0
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)
    
    # Plucked string
    u0 = np.piecewise(x, [x < 0.5, x >= 0.5], [lambda x: 2*x, lambda x: 2*(1-x)])
    v0 = np.zeros_like(u0) # Start from rest
    
    c = 1.0
    dt = 0.005 # C = c * dt / dx = 1.0 * 0.005 / 0.01 = 0.5 < 1.0 (stable)
    
    hist = wave_1d(u0, v0, c, dt, dx, n_steps=200)
    print(f"Wave equation simulated for 200 steps.")
    print(f"Midpoint value at t=0: {hist[0][nx//2]:.4f}")
    print(f"Midpoint value at t=200: {hist[-1][nx//2]:.4f}")
