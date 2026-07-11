"""
Chapter 6: Physics-Informed Neural Networks
Inverse PINN

Recover unknown thermal conductivity 'alpha' from sparse sensor measurements
of the heat equation.
"""

import torch
import torch.nn as nn

class InversePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1)
        )
        # Unknown parameter to discover
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

def exact_heat_solution(x, t, alpha_true):
    return torch.sin(torch.pi * x) * torch.exp(-(torch.pi**2) * alpha_true * t)

def main():
    print("Training Inverse PINN for Parameter Discovery...")
    torch.manual_seed(42)
    
    alpha_true = 0.1
    model = InversePINN()
    
    # Sparse sensor data (supervised data)
    N_data = 100
    x_data = torch.empty(N_data, 1).uniform_(0, 1)
    t_data = torch.empty(N_data, 1).uniform_(0, 1)
    u_data = exact_heat_solution(x_data, t_data, alpha_true)
    
    # Collocation points for PDE
    N_f = 2000
    x_f = torch.empty(N_f, 1).uniform_(0, 1)
    t_f = torch.empty(N_f, 1).uniform_(0, 1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    epochs = 3000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Data loss
        u_pred_data = model(x_data, t_data)
        loss_data = torch.mean((u_pred_data - u_data)**2)
        
        # PDE loss
        x_f.requires_grad_(True)
        t_f.requires_grad_(True)
        u_f = model(x_f, t_f)
        u_t = torch.autograd.grad(u_f, t_f, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
        u_x = torch.autograd.grad(u_f, x_f, grad_outputs=torch.ones_like(u_f), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        residual = u_t - model.alpha * u_xx
        loss_f = torch.mean(residual**2)
        
        loss = loss_data + loss_f
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Alpha true: {alpha_true} | Alpha pred: {model.alpha.item():.4f}")
            
    print(f"Inverse problem solved. Discovered alpha: {model.alpha.item():.5f} (Error: {abs(model.alpha.item() - alpha_true):.5f})")

if __name__ == "__main__":
    main()
