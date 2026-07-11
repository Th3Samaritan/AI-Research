"""
Chapter 10: Neural Operator for Temperature Fields
Simplified 1D DeepONet / Neural Operator structure.
Note: Requires PyTorch.
"""
import torch
import torch.nn as nn
import numpy as np

class BranchNet(nn.Module):
    """Processes the input function (e.g., thermal conductivity distribution)"""
    def __init__(self, sensors_dim, p=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sensors_dim, 32),
            nn.ReLU(),
            nn.Linear(32, p)
        )
    def forward(self, x):
        return self.net(x)

class TrunkNet(nn.Module):
    """Processes the evaluation coordinates (x)"""
    def __init__(self, coord_dim=1, p=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, 32),
            nn.ReLU(),
            nn.Linear(32, p)
        )
    def forward(self, x):
        return self.net(x)

class SimpleNeuralOperator(nn.Module):
    def __init__(self, sensors_dim, p=16):
        super().__init__()
        self.branch = BranchNet(sensors_dim, p)
        self.trunk = TrunkNet(1, p)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, u, y):
        """
        u: input function evaluated at sensors (Batch, Sensors)
        y: query locations (Batch, 1)
        """
        b_out = self.branch(u) # (Batch, p)
        t_out = self.trunk(y)  # (Batch, p)
        
        # Dot product
        out = torch.sum(b_out * t_out, dim=1, keepdim=True) + self.bias
        return out

def demo_neural_operator():
    print("--- Neural Operator for Parameterized PDEs ---")
    
    # We want to map a family of material conductivities k(x) 
    # to the resulting temperature field T(x)
    
    n_sensors = 20
    batch_size = 5
    
    model = SimpleNeuralOperator(sensors_dim=n_sensors, p=16)
    
    # Dummy input data: 5 different material samples
    # For each sample, we have conductivity evaluated at 20 sensors
    k_inputs = torch.rand((batch_size, n_sensors))
    
    # We want to predict temperature at random query points y in [0, 1]
    y_queries = torch.rand((batch_size, 1))
    
    # Forward pass through Operator
    T_preds = model(k_inputs, y_queries)
    
    print(f"Input functions shape (Batch, Sensors): {k_inputs.shape}")
    print(f"Query points shape (Batch, Dim): {y_queries.shape}")
    print(f"Operator output shape (Batch, 1): {T_preds.shape}")
    
    print("\nIn a full training loop, this operator learns the mapping k(x) -> T(x)")
    print("without needing to retrain for new material configurations!")
    
    print("\nSample predictions (untrained):")
    for i in range(batch_size):
        print(f"Sample {i}: Query y={y_queries[i,0].item():.3f} -> Predicted T={T_preds[i,0].item():.4f}")

if __name__ == "__main__":
    demo_neural_operator()
