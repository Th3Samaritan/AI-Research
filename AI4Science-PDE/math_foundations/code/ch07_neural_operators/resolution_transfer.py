"""
Chapter 7: Neural Operators
Resolution Transfer in FNO

Demonstrates zero-shot super-resolution by training an FNO 
at N=64 grid points and evaluating it at N=256 grid points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import FNO1d from local module
from fno_1d import FNO1d

def generate_data(resolution, batch_size):
    x = torch.linspace(0, 1, resolution)
    # Generate simple smooth functions
    k = torch.randint(1, 4, (batch_size, 1))
    a = torch.randn(batch_size, 1)
    
    # Input function: a * sin(k*pi*x)
    inputs = a * torch.sin(k * torch.pi * x.unsqueeze(0))
    
    # Target function (e.g. anti-derivative): -(a/(k*pi)) * cos(k*pi*x)
    targets = -(a / (k * torch.pi)) * torch.cos(k * torch.pi * x.unsqueeze(0))
    
    # Add spatial grid to inputs
    x_grid = x.unsqueeze(0).repeat(batch_size, 1)
    inputs_with_grid = torch.stack([inputs, x_grid], dim=-1) # [batch, res, 2]
    
    return inputs_with_grid, targets.unsqueeze(-1)

def main():
    print("Testing FNO Resolution Transfer...")
    torch.manual_seed(123)
    
    # Train at resolution 64
    res_train = 64
    model = FNO1d(modes=8, width=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Generate train data
    X_train, Y_train = generate_data(res_train, 1000)
    
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = nn.MSELoss()(pred, Y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} (Res 64) | Loss: {loss.item():.6f}")
            
    print("Training complete. Evaluating zero-shot on resolution 256...")
    
    # Evaluate at resolution 256
    res_eval = 256
    X_test, Y_test = generate_data(res_eval, 100)
    
    model.eval()
    with torch.no_grad():
        pred_high_res = model(X_test)
        test_loss = nn.MSELoss()(pred_high_res, Y_test)
        
    print(f"High-Resolution (256) Evaluation MSE Loss: {test_loss.item():.6f}")
    if test_loss.item() < 0.05:
        print("Resolution transfer successful!")
    else:
        print("Loss is high, check model convergence.")

if __name__ == "__main__":
    main()
