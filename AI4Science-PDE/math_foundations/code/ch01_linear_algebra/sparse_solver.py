"""
Chapter 1: Sparse Solvers
Sparse Laplacian assembly and Conjugate Gradient Solver.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def assemble_1d_laplacian(nx):
    """
    Assemble the 1D discrete Laplacian matrix using sparse formats.
    """
    diags = [-1 * np.ones(nx-1), 2 * np.ones(nx), -1 * np.ones(nx-1)]
    A = sp.diags(diags, offsets=[-1, 0, 1], format='csr')
    return A

def assemble_2d_laplacian(nx, ny):
    """
    Assemble the 2D discrete Laplacian matrix (5-point stencil).
    """
    D1x = assemble_1d_laplacian(nx)
    D1y = assemble_1d_laplacian(ny)
    
    Ix = sp.eye(nx)
    Iy = sp.eye(ny)
    
    # Laplacian in 2D is Ix kron D1y + D1x kron Iy
    A = sp.kron(Iy, D1x) + sp.kron(D1y, Ix)
    return A

def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    Conjugate Gradient method for solving Ax = b for SPD matrix A.
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
        
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)
    
    res_history = [np.sqrt(rs_old)]
    
    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        
        res_history.append(np.sqrt(rs_new))
        
        if np.sqrt(rs_new) < tol:
            print(f"CG converged in {i+1} iterations.")
            break
            
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
        
    return x, res_history

if __name__ == "__main__":
    print("--- 1D Laplacian ---")
    nx = 5
    A_1d = assemble_1d_laplacian(nx)
    print("Dense representation of 1D Laplacian:")
    print(A_1d.toarray())
    print()
    
    print("--- Conjugate Gradient ---")
    nx, ny = 20, 20
    N = nx * ny
    A_2d = assemble_2d_laplacian(nx, ny)
    
    # Random RHS
    b = np.random.rand(N)
    
    # Solve using custom CG
    x, res_hist = conjugate_gradient(A_2d, b)
    
    # Verify with scipy
    x_sp, _ = spla.cg(A_2d, b, tol=1e-8)
    
    print(f"Error compared to SciPy: {np.linalg.norm(x - x_sp)}")
