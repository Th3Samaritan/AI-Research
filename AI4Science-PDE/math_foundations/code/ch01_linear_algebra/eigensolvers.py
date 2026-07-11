"""
Chapter 1: Eigensolvers
Implementation of Power Iteration, QR Algorithm, and SVD via Eigen-decomposition.
"""
import numpy as np

def power_iteration(A, num_simulations=100, tol=1e-8):
    """
    Find the dominant eigenvalue and corresponding eigenvector.
    """
    n = A.shape[0]
    b_k = np.random.rand(n)
    
    for _ in range(num_simulations):
        b_k1 = A @ b_k
        b_k1_norm = np.linalg.norm(b_k1)
        b_k_next = b_k1 / b_k1_norm
        if np.linalg.norm(b_k_next - b_k) < tol:
            b_k = b_k_next
            break
        b_k = b_k_next
        
    eigenvalue = (b_k.T @ A @ b_k) / (b_k.T @ b_k)
    return eigenvalue, b_k

def qr_algorithm(A, num_iter=100):
    """
    Find all eigenvalues of a matrix using the QR algorithm.
    """
    Ak = A.copy()
    for _ in range(num_iter):
        Q, R = np.linalg.qr(Ak) # Using numpy's more stable QR for iterations
        Ak = R @ Q
    return np.diag(Ak)

def svd_via_eigen(A):
    """
    Compute SVD using eigen-decomposition of A^T A and A A^T.
    Returns U, S, V^T such that A = U @ diag(S) @ V^T.
    """
    # V^T from A^T A
    AtA = A.T @ A
    eigenvalues_v, V = np.linalg.eigh(AtA)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues_v)[::-1]
    eigenvalues_v = eigenvalues_v[idx]
    V = V[:, idx]
    
    # Singular values (take only the top min(m, n))
    k = min(A.shape)
    S = np.sqrt(np.maximum(eigenvalues_v[:k], 0))
    
    # U from A A^T
    AAt = A @ A.T
    eigenvalues_u, U = np.linalg.eigh(AAt)
    idx_u = np.argsort(eigenvalues_u)[::-1]
    U = U[:, idx_u]
    
    # Adjust signs of U
    for i in range(k):
        if S[i] > 1e-10:
            u_expected = (A @ V[:, i]) / S[i]
            if np.dot(u_expected, U[:, i]) < 0:
                U[:, i] = -U[:, i]
                
    return U, S, V.T

if __name__ == "__main__":
    print("--- Power Iteration ---")
    A = np.array([[2, -12], [1, -5]])
    val, vec = power_iteration(A)
    print("Dominant Eigenvalue:", val)
    print("Eigenvector:", vec)
    print()
    
    print("--- QR Algorithm ---")
    A_sym = np.array([[4, 1, -2], [1, 2, 0], [-2, 0, 3]])
    vals = qr_algorithm(A_sym)
    print("Matrix A_sym:\n", A_sym)
    print("Eigenvalues (QR Algo):", vals)
    print("Eigenvalues (numpy):", np.linalg.eigvals(A_sym))
    print()
    
    print("--- SVD via Eigen-decomposition ---")
    M = np.array([[3, 2, 2], [2, 3, -2]])
    U, S, Vt = svd_via_eigen(M)
    print("M:\n", M)
    print("U:\n", U)
    print("S:\n", S)
    print("V^T:\n", Vt)
    
    S_mat = np.zeros((U.shape[0], Vt.shape[0]))
    np.fill_diagonal(S_mat, S)
    print("Reconstructed M:\n", U @ S_mat @ Vt)
