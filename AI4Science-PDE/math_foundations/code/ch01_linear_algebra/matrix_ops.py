"""
Chapter 1: Matrix Operations from Scratch
This module implements fundamental matrix factorizations:
LU, Cholesky, and QR (via Gram-Schmidt).
"""
import numpy as np

def lu_decomposition(A):
    """
    Perform LU decomposition of a square matrix A without pivoting.
    Returns L, U such that A = L @ U.
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    
    for k in range(n-1):
        for i in range(k+1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
            
    return L, U

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition of a symmetric, positive-definite matrix A.
    Returns L such that A = L @ L.T.
    """
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    
    for i in range(n):
        for j in range(i+1):
            sum_k = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - sum_k)
            else:
                L[i, j] = (A[i, j] - sum_k) / L[j, j]
                
    return L

def qr_gram_schmidt(A):
    """
    Perform QR decomposition using the Classical Gram-Schmidt process.
    Returns Q, R such that A = Q @ R, where Q is orthogonal and R is upper triangular.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].astype(float)
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-12:
            Q[:, j] = v / R[j, j]
            
    return Q, R

if __name__ == "__main__":
    print("--- LU Decomposition ---")
    A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
    L, U = lu_decomposition(A)
    print("A:\n", A)
    print("L:\n", L)
    print("U:\n", U)
    print("L @ U:\n", L @ U)
    print()
    
    print("--- Cholesky Decomposition ---")
    B = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    L_chol = cholesky_decomposition(B)
    print("B:\n", B)
    print("L:\n", L_chol)
    print("L @ L.T:\n", L_chol @ L_chol.T)
    print()
    
    print("--- QR Decomposition (Gram-Schmidt) ---")
    C = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
    Q, R = qr_gram_schmidt(C)
    print("C:\n", C)
    print("Q:\n", Q)
    print("R:\n", R)
    print("Q @ R:\n", Q @ R)
