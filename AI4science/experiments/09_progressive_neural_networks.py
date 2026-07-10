import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION & REPRODUCIBILITY
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==========================================
# 1. NON-STATIONARY FUZZY ENVIRONMENT
# ==========================================
class ConceptDriftEnvironment:
    def generate_batch(self, batch_size=32, mode=0):
        X = torch.rand(batch_size, 5) * 4.0 
        if mode == 0:
            X[:, 4] = torch.normal(mean=-2.0, std=0.5, size=(batch_size,))
            Y = (X[:, 0] * X[:, 1]) + torch.sin(X[:, 2]) - X[:, 3]
        else:
            X[:, 4] = torch.normal(mean=2.0, std=0.5, size=(batch_size,))
            Y = (X[:, 2] ** 2) - torch.cos(X[:, 0] * X[:, 1]) + X[:, 3]
            
        Y += torch.normal(mean=0.0, std=0.2, size=(batch_size,))
        return X, Y.unsqueeze(1)

# ==========================================
# 2. ARCHITECTURES
# ==========================================
class StandardMLP(nn.Module):
    """The Baseline Monolithic Network"""
    def __init__(self):
        super(StandardMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------------------
# Dynamic Structural Expansion Modules
# ------------------------------------------
class PNNColumn(nn.Module):
    """A single neural column dedicated to one specific task/mode."""
    def __init__(self, num_prev_columns):
        super(PNNColumn, self).__init__()
        self.layer1 = nn.Linear(5, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
        self.num_prev_columns = num_prev_columns
        
        # Lateral connections reading from frozen previous columns
        if num_prev_columns > 0:
            self.lat1 = nn.Linear(32 * num_prev_columns, 32)
            self.lat2 = nn.Linear(32 * num_prev_columns, 32)

    def forward(self, x, prev_h1=[], prev_h2=[]):
        h1 = self.relu(self.layer1(x))
        if self.num_prev_columns > 0:
            lat_in1 = torch.cat(prev_h1, dim=1)
            h1 = h1 + self.relu(self.lat1(lat_in1))
            
        h2 = self.relu(self.layer2(h1))
        if self.num_prev_columns > 0:
            lat_in2 = torch.cat(prev_h2, dim=1)
            h2 = h2 + self.relu(self.lat2(lat_in2))
            
        out = self.layer3(h2)
        return out, h1, h2

class ProgressiveNeuralNetwork(nn.Module):
    """Dynamically expanding network that solves Catastrophic Forgetting."""
    def __init__(self):
        super(ProgressiveNeuralNetwork, self).__init__()
        # Starts with just 1 column for the first universe
        self.columns = nn.ModuleList([PNNColumn(0)])
        
    def freeze_all(self):
        """Mathematical lockdown of historical memories."""
        for param in self.parameters():
            param.requires_grad = False
            
    def add_column(self):
        """Physical expansion of the neural architecture."""
        self.freeze_all()
        self.columns.append(PNNColumn(len(self.columns)))
        
    def forward(self, x, task_id):
        # Propagates through requested task and its lateral dependencies
        prev_h1, prev_h2 = [], []
        out = None
        for i in range(task_id + 1):
            out, h1, h2 = self.columns[i](x, prev_h1, prev_h2)
            prev_h1.append(h1)
            prev_h2.append(h2)
        return out

# ==========================================
# 3. EXPERIMENT EXECUTION
# ==========================================
def run_structural_expansion_proof():
    env = ConceptDriftEnvironment()
    
    model_A = StandardMLP()
    model_B = ProgressiveNeuralNetwork()
    
    # Track the active parameters to show the hardware cost
    def count_active_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt_A = optim.Adam(model_A.parameters(), lr=0.005)
    opt_B = optim.Adam(filter(lambda p: p.requires_grad, model_B.parameters()), lr=0.005)
    mse_loss = nn.MSELoss()
    
    epochs = 400
    acc_A_mode0, acc_A_mode1 = [], []
    acc_B_mode0, acc_B_mode1 = [], []
    epochs_logged = []
    
    print("--- STARTING DYNAMIC STRUCTURAL EXPANSION PROOF ---")
    
    for epoch in range(epochs):
        # ----------------------------------------------------
        # THE CONCEPT DRIFT & STRUCTURAL EXPANSION
        # ----------------------------------------------------
        current_training_mode = 0 if epoch < 200 else 1
        X_train, Y_train = env.generate_batch(batch_size=64, mode=current_training_mode)
        
        if epoch == 200:
            print("\n[Concept Drift Triggered] Model B is initiating Structural Expansion...")
            print(f"Model B Params BEFORE Expansion: {count_active_params(model_B)}")
            
            # Physically grow Model B to accommodate Mode 1
            model_B.add_column()
            
            # Rebind optimizer to only train the newly spawned column
            opt_B = optim.Adam(filter(lambda p: p.requires_grad, model_B.parameters()), lr=0.005)
            
            print(f"Model B Freezing Task 0... Spawning Task 1 Column.")
            print(f"Model B Active Training Params AFTER Expansion: {count_active_params(model_B)}\n")

        # --- Train Model A (Static MLP) ---
        opt_A.zero_grad()
        pred_A = model_A(X_train)
        loss_A = mse_loss(pred_A, Y_train)
        loss_A.backward()
        opt_A.step()
        
        # --- Train Model B (Progressive Network) ---
        opt_B.zero_grad()
        # Train strictly on the current task's column
        pred_B = model_B(X_train, task_id=current_training_mode)
        loss_B = mse_loss(pred_B, Y_train)
        loss_B.backward()
        opt_B.step()
        
        # --- Evaluation Tracking ---
        if epoch % 10 == 0:
            epochs_logged.append(epoch)
            X_test0, Y_test0 = env.generate_batch(100, mode=0)
            X_test1, Y_test1 = env.generate_batch(100, mode=1)
            
            def eval_A(X, Y):
                with torch.no_grad():
                    return ((torch.abs(model_A(X) - Y) < 0.75).sum().item() / len(Y)) * 100
                    
            def eval_B(X, Y, task_id):
                with torch.no_grad():
                    return ((torch.abs(model_B(X, task_id) - Y) < 0.75).sum().item() / len(Y)) * 100

            acc_A_mode0.append(eval_A(X_test0, Y_test0))
            acc_A_mode1.append(eval_A(X_test1, Y_test1))
            
            acc_B_mode0.append(eval_B(X_test0, Y_test0, task_id=0))
            acc_B_mode1.append(eval_B(X_test1, Y_test1, task_id=1 if epoch >= 200 else 0))

    # ==========================================
    # RENDER SCIENTIFIC METRICS
    # ==========================================
    print("Training Complete. Rendering Absolute Isolation Proof...")
    
    plt.figure(figsize=(12, 7))
    plt.style.use('bmh')
    
    # Baseline Model A
    plt.plot(epochs_logged, acc_A_mode0, color='#d62728', linestyle='-', linewidth=2, alpha=0.6, label='Model A (Static MLP) - Mode 0')
    plt.plot(epochs_logged, acc_A_mode1, color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.6, label='Model A (Static MLP) - Mode 1')
    
    # Progressive Model B
    plt.plot(epochs_logged, acc_B_mode0, color='#1f77b4', linestyle='-', linewidth=3, label='Model B (Progressive Net) - Mode 0')
    plt.plot(epochs_logged, acc_B_mode1, color='#2ca02c', linestyle='--', linewidth=3, label='Model B (Progressive Net) - Mode 1')
    
    plt.axvline(x=200, color='black', linestyle='-.', linewidth=2, label='Concept Drift & Dynamic Expansion')
    
    plt.title('Absolute Continuous Learning via Dynamic Structural Expansion', fontsize=14, pad=15)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Fuzzy Accuracy (%)', fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.ylim(0, 105)
    plt.subplots_adjust(right=0.7)
    plt.grid(True, alpha=0.6)
    
    plt.show()

if __name__ == "__main__":
    run_structural_expansion_proof()