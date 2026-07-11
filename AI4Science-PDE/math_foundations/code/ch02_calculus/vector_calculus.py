"""
Chapter 2: Vector Calculus on Grids
Numerical gradient, divergence, curl, and Laplacian using central differences.
"""
import numpy as np

def gradient(F, dx, dy):
    """Compute 2D numerical gradient of a scalar field F."""
    dFdy, dFdx = np.gradient(F, dy, dx)
    return dFdx, dFdy

def divergence(U, V, dx, dy):
    """Compute 2D divergence of a vector field (U, V)."""
    dUdx = np.gradient(U, dx, axis=1)
    dVdy = np.gradient(V, dy, axis=0)
    return dUdx + dVdy

def curl_2d(U, V, dx, dy):
    """Compute 2D curl (z-component) of a vector field (U, V)."""
    dUdy = np.gradient(U, dy, axis=0)
    dVdx = np.gradient(V, dx, axis=1)
    return dVdx - dUdy

def laplacian(F, dx, dy):
    """Compute 2D Laplacian of a scalar field F."""
    d2Fdx2 = np.gradient(np.gradient(F, dx, axis=1), dx, axis=1)
    d2Fdy2 = np.gradient(np.gradient(F, dy, axis=0), dy, axis=0)
    return d2Fdx2 + d2Fdy2

if __name__ == "__main__":
    print("--- Vector Calculus on Grids ---")
    # Create a grid
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Scalar field F = exp(-(X^2 + Y^2))
    F = np.exp(-(X**2 + Y**2))
    
    grad_x, grad_y = gradient(F, dx, dy)
    print("Gradient shape:", grad_x.shape)
    
    # Vector field U = -Y, V = X
    U = -Y
    V = X
    
    div = divergence(U, V, dx, dy)
    print("Divergence of (-y, x): min/max =", np.min(div), np.max(div))
    
    curl = curl_2d(U, V, dx, dy)
    print("Curl of (-y, x): min/max =", np.min(curl), np.max(curl)) # Should be approx 2
    
    lapF = laplacian(F, dx, dy)
    print("Laplacian shape:", lapF.shape)
    print("Vector calculus operations successful.")
