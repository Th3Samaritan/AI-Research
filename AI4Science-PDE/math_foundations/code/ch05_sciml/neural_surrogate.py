"""
Chapter 5: Scientific Machine Learning
Neural Surrogate for a Parameterized PDE Solution

This module trains a Multi-Layer Perceptron (MLP) to approximate 
the solution to a parameterized 1D heat equation:
u(x, t; a) = sin(a * pi * x) * exp(-(a * pi)^2 * t)
where 'a' is a parameter.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class NeuralSurrogate(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), # inputs: x, t, a
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def exact_solution(x, t, a):
    return torch.sin(a * torch.pi * x) * torch.exp(-(a * torch.pi)**2 * t)

def main():
    print("Training Neural Surrogate for Parameterized PDE...")
    torch.manual_seed(42)
    
    # Generate Training Data
    N_train = 5000
    x_train = torch.rand(N_train, 1)
    t_train = torch.rand(N_train, 1) * 0.1
    a_train = torch.rand(N_train, 1) * 2 + 1 # a in [1, 3]
    
    X_train = torch.cat([x_train, t_train, a_train], dim=1)
    y_train = exact_solution(x_train, t_train, a_train)
    
    model = NeuralSurrogate()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training Loop
    epochs = 2000
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
    print("Training complete. Testing generalization...")
    
    # Test Generalization on unseen parameter 'a'
    a_test = 2.5 # Unseen specifically, though in range
    x_test = torch.linspace(0, 1, 100).unsqueeze(1)
    t_test = torch.full((100, 1), 0.05)
    a_test_tensor = torch.full((100, 1), a_test)
    
    X_test = torch.cat([x_test, t_test, a_test_tensor], dim=1)
    y_test_exact = exact_solution(x_test, t_test, a_test_tensor)
    
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        
    test_mse = criterion(y_test_pred, y_test_exact).item()
    print(f"Generalization Test MSE: {test_mse:.6f}")

if __name__ == "__main__":
    main()
