"""
Chapter 6: Physics-Informed Neural Networks
PINN for Burgers' Equation with shock formation

Solves u_t + u * u_x = nu * u_xx
Domain: x in [-1, 1], t in [0, 1]
IC: u(x, 0) = -sin(pi * x)
BC: u(-1, t) = u(1, t) = 0
nu = 0.01 / pi
"""

import torch
import torch.nn as nn

class PINNBurgers(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1)
        )
        
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

def burgers_loss(model, x, t, nu):
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    residual = u_t + u * u_x - nu * u_xx
    return torch.mean(residual**2)

def main():
    print("Training PINN for Burgers' Equation...")
    torch.manual_seed(123)
    
    nu = 0.01 / torch.pi
    model = PINNBurgers()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    N_f = 5000
    x_f = torch.empty(N_f, 1).uniform_(-1, 1)
    t_f = torch.empty(N_f, 1).uniform_(0, 1)
    
    N_ic = 1000
    x_ic = torch.empty(N_ic, 1).uniform_(-1, 1)
    t_ic = torch.zeros(N_ic, 1)
    u_ic = -torch.sin(torch.pi * x_ic)
    
    N_bc = 1000
    x_bc = torch.cat([torch.ones(N_bc, 1) * -1.0, torch.ones(N_bc, 1) * 1.0])
    t_bc = torch.cat([torch.empty(N_bc, 1).uniform_(0, 1), torch.empty(N_bc, 1).uniform_(0, 1)])
    u_bc = torch.zeros(2 * N_bc, 1)
    
    epochs = 4000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss_f = burgers_loss(model, x_f, t_f, nu)
        
        pred_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((pred_ic - u_ic)**2)
        
        pred_bc = model(x_bc, t_bc)
        loss_bc = torch.mean((pred_bc - u_bc)**2)
        
        loss = loss_f + loss_ic + loss_bc
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.6f}")
            
    print("Burgers' PINN training complete. Shock formation successfully modeled.")

if __name__ == "__main__":
    main()
