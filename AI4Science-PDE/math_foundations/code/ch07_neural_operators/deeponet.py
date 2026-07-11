"""
Chapter 7: Neural Operators
DeepONet from scratch

Learning a 1D integral operator (anti-derivative):
G(u)(y) = int_0^y u(x) dx
"""

import torch
import torch.nn as nn

class DeepONet(nn.Module):
    def __init__(self, num_sensors, p=40):
        super().__init__()
        # Branch network takes the function evaluated at sensors
        self.branch = nn.Sequential(
            nn.Linear(num_sensors, 64),
            nn.ReLU(),
            nn.Linear(64, p)
        )
        
        # Trunk network takes the continuous evaluation point y
        self.trunk = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, p)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, u_sensors, y):
        # u_sensors: [batch, num_sensors]
        # y: [batch, 1]
        branch_out = self.branch(u_sensors) # [batch, p]
        trunk_out = self.trunk(y)           # [batch, p]
        
        # Dot product
        output = torch.sum(branch_out * trunk_out, dim=-1, keepdim=True) + self.bias
        return output

def main():
    print("Training DeepONet for Anti-Derivative Operator...")
    torch.manual_seed(0)
    
    num_sensors = 100
    x_sensors = torch.linspace(0, 1, num_sensors)
    
    # Generate data: u(x) = a * sin(k * pi * x)
    # G(u)(y) = (a / (k * pi)) * (1 - cos(k * pi * y))
    
    batch_size = 2000
    a = torch.rand(batch_size, 1) * 2
    k = torch.randint(1, 5, (batch_size, 1)).float()
    
    # Input functions evaluated at sensors
    u_train = a * torch.sin(k * torch.pi * x_sensors.unsqueeze(0))
    
    # Random evaluation points y
    y_train = torch.rand(batch_size, 1)
    
    # Exact operator output
    G_train = (a / (k * torch.pi)) * (1 - torch.cos(k * torch.pi * y_train))
    
    model = DeepONet(num_sensors)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(u_train, y_train)
        loss = criterion(pred, G_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
            
    print("DeepONet training complete.")

if __name__ == "__main__":
    main()
