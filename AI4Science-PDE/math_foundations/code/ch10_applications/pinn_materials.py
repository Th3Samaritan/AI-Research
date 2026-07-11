"""
Chapter 10: PINN for Inverse Thermal Conductivity Identification
Note: Requires PyTorch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Network to approximate Temperature T(x)
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Learnable parameter for unknown thermal conductivity
        self.k = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return self.net(x)

def train_pinn():
    print("--- PINN for Inverse Material Property Discovery ---")
    
    # True problem: 1D heat equation, d/dx(k dT/dx) = -Q
    # Domain: x in [0, 1]
    # Let true k = 3.5
    # Let Q = 2.0
    # BCs: T(0) = 0, T(1) = 0
    # True solution: T(x) = (Q/2k) * x * (1 - x) = (2 / 7) * x * (1 - x)
    
    true_k = 3.5
    Q_source = 2.0
    
    # Generate noisy observation data
    x_data = np.linspace(0, 1, 20).reshape(-1, 1)
    T_exact = (Q_source / (2 * true_k)) * x_data * (1 - x_data)
    noise = np.random.normal(0, 0.01, T_exact.shape)
    T_obs = T_exact + noise
    
    x_tensor = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
    T_tensor = torch.tensor(T_obs, dtype=torch.float32)
    
    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"True conductivity k: {true_k}")
    print(f"Initial PINN k guess: {model.k.item():.4f}")
    
    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Data Loss (fit the observations)
        T_pred = model(x_tensor)
        loss_data = torch.mean((T_pred - T_tensor)**2)
        
        # 2. Physics Loss (satisfy the PDE)
        # T_x = dT/dx
        T_x = torch.autograd.grad(T_pred, x_tensor, 
                                  grad_outputs=torch.ones_like(T_pred),
                                  create_graph=True)[0]
        # T_xx = d2T/dx2
        T_xx = torch.autograd.grad(T_x, x_tensor, 
                                   grad_outputs=torch.ones_like(T_x),
                                   create_graph=True)[0]
                                   
        # PDE: k * T_xx + Q = 0
        pde_residual = model.k * T_xx + Q_source
        loss_physics = torch.mean(pde_residual**2)
        
        # 3. Total Loss
        loss = loss_data + 1e-4 * loss_physics
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss.item():.6f}, Estimated k = {model.k.item():.4f}")
            
    print("\nTraining complete.")
    print(f"Final Estimated conductivity k: {model.k.item():.4f} (Target: {true_k})")
    error = abs(model.k.item() - true_k) / true_k * 100
    print(f"Error: {error:.2f}%")

if __name__ == "__main__":
    train_pinn()
