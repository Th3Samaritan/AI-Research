"""
Chapter 10: 2D Steady-State Heat Conduction through Composite Material
Finite Difference Solver.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import os

def solve_composite_heat(nx, ny):
    print("Setting up composite material domain...")
    
    # Thermal conductivities
    k_background = 1.0
    k_inclusion = 50.0
    
    # Grid properties
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    # Build thermal conductivity field
    K = np.ones((nx, ny)) * k_background
    
    # Add a highly conductive inclusion in the center
    cx, cy, r = nx // 2, ny // 2, nx // 5
    for i in range(nx):
        for j in range(ny):
            if (i - cx)**2 + (j - cy)**2 <= r**2:
                K[i, j] = k_inclusion
                
    # Build sparse matrix for the system: -div(K grad T) = 0
    N = nx * ny
    
    def idx(i, j):
        return i * ny + j
        
    print("Constructing matrix system...")
    # Better approach for variable coefficients: LIL matrix
    A = sp.lil_matrix((N, N))
    b = np.zeros(N)
    
    for i in range(nx):
        for j in range(ny):
            curr = idx(i, j)
            if i == 0:
                A[curr, curr] = 1.0
                b[curr] = 100.0
            elif i == nx - 1:
                A[curr, curr] = 1.0
                b[curr] = 0.0
            elif j == 0:
                A[curr, curr] = 1.0
                A[curr, idx(i, j+1)] = -1.0
                b[curr] = 0.0
            elif j == ny - 1:
                A[curr, curr] = 1.0
                A[curr, idx(i, j-1)] = -1.0
                b[curr] = 0.0
            else:
                kx_plus = 2 * K[i,j] * K[i+1,j] / (K[i,j] + K[i+1,j])
                kx_minus = 2 * K[i,j] * K[i-1,j] / (K[i,j] + K[i-1,j])
                ky_plus = 2 * K[i,j] * K[i,j+1] / (K[i,j] + K[i,j+1])
                ky_minus = 2 * K[i,j] * K[i,j-1] / (K[i,j] + K[i,j-1])
                
                A[curr, curr] = (kx_plus + kx_minus)/dx**2 + (ky_plus + ky_minus)/dy**2
                A[curr, idx(i+1, j)] = -kx_plus/dx**2
                A[curr, idx(i-1, j)] = -kx_minus/dx**2
                A[curr, idx(i, j+1)] = -ky_plus/dy**2
                A[curr, idx(i, j-1)] = -ky_minus/dy**2
                
    print("Solving linear system...")
    A_csr = A.tocsr()
    T_flat = spla.spsolve(A_csr, b)
    T = T_flat.reshape((nx, ny))
    
    print("Solution complete. Statistics:")
    print(f"Max Temp: {np.max(T):.2f}, Min Temp: {np.min(T):.2f}")
    
    # Save a simple textual plot
    print("\nTemperature cross-section at mid-y:")
    mid_y = ny // 2
    for i in range(0, nx, nx//10):
        print(f"x={i*dx:.2f}: {T[i, mid_y]:.2f} °C (K={K[i, mid_y]:.1f})")

if __name__ == "__main__":
    solve_composite_heat(50, 50)
