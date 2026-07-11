"""
Chapter 4: Numerical Methods - Spectral Methods
FFT-based derivative computation and spectral Poisson solver.
"""
import numpy as np

def spectral_derivative(u, L):
    """
    Compute first derivative using FFT.
    Domain length L.
    """
    N = len(u)
    dx = L / N
    
    # Wavenumbers
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    
    u_hat = np.fft.fft(u)
    # Derivative in Fourier space: multiply by i*k
    du_hat = 1j * k * u_hat
    
    du = np.real(np.fft.ifft(du_hat))
    return du

def spectral_poisson_1d(f, L):
    """
    Solve 1D Poisson equation -u'' = f using Spectral Methods.
    Periodic boundary conditions assumed.
    Note: integral of f must be 0 for a periodic solution to exist.
    """
    N = len(f)
    dx = L / N
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    
    f_hat = np.fft.fft(f)
    
    u_hat = np.zeros_like(f_hat)
    # -u'' = f -> (k^2) u_hat = f_hat
    for i in range(1, N): # Skip k=0 to avoid division by zero
        u_hat[i] = f_hat[i] / (k[i]**2)
        
    u_hat[0] = 0.0 # Set mean to zero
    
    u = np.real(np.fft.ifft(u_hat))
    return u

if __name__ == "__main__":
    print("--- Spectral Methods (FFT) ---")
    N = 64
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    
    # Test derivative: u = sin(x), u' = cos(x)
    u = np.sin(x)
    du_exact = np.cos(x)
    du_spectral = spectral_derivative(u, L)
    print("Max error in spectral derivative:", np.max(np.abs(du_spectral - du_exact)))
    
    # Test Poisson: -u'' = f
    # Let true u = sin(2x). Then -u'' = 4*sin(2x)
    f = 4 * np.sin(2 * x)
    u_spectral = spectral_poisson_1d(f, L)
    
    u_exact = np.sin(2 * x)
    print("Max error in spectral Poisson solver:", np.max(np.abs(u_spectral - u_exact)))
