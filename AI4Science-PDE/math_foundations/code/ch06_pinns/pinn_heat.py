"""
Chapter 6: Physics-Informed Neural Networks
PINN for 1D Heat Equation

Solves u_t = alpha * u_xx
Domain: x in [-1, 1], t in [0, 1]
IC: u(x, 0) = -sin(pi * x)
BC: u(-1, t) = u(1, t) = 0
"""

import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1)
        )
        
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

def physics_loss(model, x, t, alpha=0.1):
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    residual = u_t - alpha * u_xx
    return torch.mean(residual**2)

def main():
    print("Training PINN for 1D Heat Equation...")
    torch.manual_seed(0)
    
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Collocation points for PDE
    N_f = 2000
    x_f = torch.empty(N_f, 1).uniform_(-1, 1)
    t_f = torch.empty(N_f, 1).uniform_(0, 1)
    
    # Initial Condition points
    N_ic = 500
    x_ic = torch.empty(N_ic, 1).uniform_(-1, 1)
    t_ic = torch.zeros(N_ic, 1)
    u_ic = -torch.sin(torch.pi * x_ic)
    
    # Boundary Condition points
    N_bc = 500
    x_bc1 = torch.ones(N_bc, 1) * -1.0
    x_bc2 = torch.ones(N_bc, 1) * 1.0
    t_bc = torch.empty(N_bc, 1).uniform_(0, 1)
    u_bc = torch.zeros(N_bc, 1)
    
    x_bc = torch.cat([x_bc1, x_bc2])
    t_bc_all = torch.cat([t_bc, t_bc])
    u_bc_all = torch.cat([u_bc, u_bc])
    
    epochs = 3000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE Loss
        loss_f = physics_loss(model, x_f, t_f)
        
        # IC Loss
        pred_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((pred_ic - u_ic)**2)
        
        # BC Loss
        pred_bc = model(x_bc, t_bc_all)
        loss_bc = torch.mean((pred_bc - u_bc_all)**2)
        
        loss = loss_f + loss_ic + loss_bc
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.6f} | PDE: {loss_f.item():.6f}")
            
    print("PINN training complete.")

if __name__ == "__main__":
    main()
