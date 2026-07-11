"""
Chapter 4: Numerical Methods - Finite Element Method (1D)
Minimal 1D FEM solver for -u'' = f with linear hat functions.
"""
import numpy as np

def assemble_fem_1d(nodes, f_func):
    """
    Assemble stiffness matrix K and load vector F for 1D Poisson -u'' = f
    using piecewise linear basis functions.
    Dirichlet BCs (u=0) on boundaries.
    """
    n_nodes = len(nodes)
    n_elems = n_nodes - 1
    
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for i in range(n_elems):
        h = nodes[i+1] - nodes[i]
        
        # Local stiffness matrix for -u''
        K_loc = np.array([[1, -1], [-1, 1]]) / h
        
        # Assemble global K
        K[i:i+2, i:i+2] += K_loc
        
        # Local load vector (using trapezoidal rule approximation)
        x_mid = 0.5 * (nodes[i] + nodes[i+1])
        f_mid = f_func(x_mid)
        F_loc = np.array([0.5, 0.5]) * h * f_mid
        
        F[i:i+2] += F_loc
        
    return K, F

def solve_fem_1d(nodes, f_func):
    """Solve the FEM system with u(0)=u(L)=0."""
    K, F = assemble_fem_1d(nodes, f_func)
    
    # Apply Dirichlet boundary conditions (zero on ends)
    # Zero out rows and columns for boundary nodes
    K[0, :] = 0; K[:, 0] = 0; K[0, 0] = 1.0; F[0] = 0.0
    K[-1, :] = 0; K[:, -1] = 0; K[-1, -1] = 1.0; F[-1] = 0.0
    
    u = np.linalg.solve(K, F)
    return u

if __name__ == "__main__":
    print("--- 1D Finite Element Method ---")
    # Domain [0, 1]
    L = 1.0
    n_nodes = 11
    nodes = np.linspace(0, L, n_nodes)
    
    # Force f(x) = 1. Exact solution: u(x) = 0.5 * x * (1 - x)
    def f_func(x):
        return 1.0
        
    u_fem = solve_fem_1d(nodes, f_func)
    u_exact = 0.5 * nodes * (1 - nodes)
    
    print("Node positions:   ", nodes)
    print("FEM Solution:     ", np.round(u_fem, 4))
    print("Exact Solution:   ", np.round(u_exact, 4))
    print("Max Error:        ", np.max(np.abs(u_fem - u_exact)))
