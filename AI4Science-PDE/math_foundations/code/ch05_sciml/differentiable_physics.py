"""
Chapter 5: Scientific Machine Learning
Differentiable Physics Solver

We embed a simple finite difference PDE solver (heat equation) into PyTorch.
Using automatic differentiation, we backpropagate through the solver steps 
to recover an unknown physical parameter (thermal diffusivity).
"""

import torch
import torch.nn as nn

def differentiable_heat_solver(u0, alpha, nx, dt, dx, steps):
    """
    1D Heat equation solver using Explicit Euler.
    u_t = alpha * u_xx
    """
    u = u0.clone()
    for _ in range(steps):
        # Finite difference for u_xx
        u_xx = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        
        # Update interior points
        u_new = u.clone()
        u_new[1:-1] = u[1:-1] + dt * alpha * u_xx
        
        # Dirichlet boundary conditions (fixed at 0)
        u_new[0] = 0.0
        u_new[-1] = 0.0
        
        u = u_new
    return u

def main():
    print("Differentiable Physics Parameter Estimation...")
    
    # Grid setup
    nx = 50
    dx = 1.0 / (nx - 1)
    x = torch.linspace(0, 1, nx)
    dt = 0.0001
    steps = 1000
    
    # Initial condition: u(x,0) = sin(pi*x)
    u0 = torch.sin(torch.pi * x)
    
    # Ground truth parameter and data generation
    alpha_true = torch.tensor(0.1)
    with torch.no_grad():
        u_target = differentiable_heat_solver(u0, alpha_true, nx, dt, dx, steps)
        
    # Optimization to find alpha
    alpha_guess = nn.Parameter(torch.tensor(0.01))
    optimizer = torch.optim.Adam([alpha_guess], lr=0.01)
    
    print(f"Initial guess for alpha: {alpha_guess.item():.4f}")
    print(f"True alpha: {alpha_true.item():.4f}")
    
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass through the differentiable solver
        u_pred = differentiable_heat_solver(u0, alpha_guess, nx, dt, dx, steps)
        
        # Compute loss against target observation
        loss = torch.mean((u_pred - u_target)**2)
        
        # Backpropagate through time (through the solver steps)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | Alpha: {alpha_guess.item():.5f}")
            
    print("Optimization finished.")
    print(f"Recovered alpha: {alpha_guess.item():.4f} (Error: {abs(alpha_guess.item() - alpha_true.item()):.4f})")

if __name__ == "__main__":
    main()
