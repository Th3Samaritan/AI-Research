"""
Chapter 8: SciPy Toolkit - Sparse Matrices, ODEs, and Optimization
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def demo_sparse_matrix():
    print("--- Sparse Matrix Construction ---")
    N = 5
    # Construct 1D finite difference matrix (Laplacian)
    diagonals = [np.ones(N-1), -2*np.ones(N), np.ones(N-1)]
    A = sp.diags(diagonals, offsets=[-1, 0, 1], format='csr')
    print("Dense representation:")
    print(A.toarray())
    print("\nSparse matrix properties:")
    print(f"Non-zeros: {A.nnz}")
    print(f"Shape: {A.shape}\n")
    
    # Solve linear system Ax = b
    b = np.ones(N)
    x = spla.spsolve(A, b)
    print(f"Solution to Ax=b: {x}\n")

def decay_model(t, y, k):
    return -k * y

def demo_ode_solver():
    print("--- ODE Solver (scipy.integrate.solve_ivp) ---")
    y0 = [1.0]
    t_span = (0, 5)
    t_eval = np.linspace(0, 5, 20)
    k = 0.5
    
    sol = solve_ivp(decay_model, t_span, y0, args=(k,), t_eval=t_eval)
    print(f"Solved ODE successfully? {sol.success}")
    print(f"Time points: {sol.t[:5]}...")
    print(f"Solution values: {sol.y[0][:5]}...\n")

def objective_function(x):
    # Rosenbrock function
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def demo_optimization():
    print("--- Inverse Problem / Optimization (scipy.optimize) ---")
    x0 = np.array([0.0, 0.0])
    res = minimize(objective_function, x0, method='BFGS')
    print("Minimizing Rosenbrock function...")
    print(f"Success: {res.success}")
    print(f"Minimum found at: {res.x}")
    print(f"Objective value: {res.fun}\n")

if __name__ == "__main__":
    demo_sparse_matrix()
    demo_ode_solver()
    demo_optimization()
