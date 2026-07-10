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
    def __init__(self):
        super(StandardMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class RecurrentRLReasoner(nn.Module):
    def __init__(self):
        super(RecurrentRLReasoner, self).__init__()
        self.rag_memory = nn.Embedding(2, 16) 
        self.state_encoder = nn.Linear(5, 16)
        self.gru = nn.GRUCell(input_size=16, hidden_size=32)
        
        self.policy = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3) 
        )
        self.predictor = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward_trajectory(self, x, max_steps=3):
        batch_size = x.size(0)
        encoded_x = self.state_encoder(x)
        h = torch.zeros(batch_size, 32)
        log_probs = []
        
        for step in range(max_steps):
            h = self.gru(encoded_x, h)
            action_logits = self.policy(h)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            log_probs.append(action_dist.log_prob(action))
            
            majority_action = torch.mode(action).values.item()
            if majority_action == 2:
                break
            elif majority_action in [0, 1]:
                retrieved_knowledge = self.rag_memory(torch.tensor([majority_action] * batch_size))
                encoded_x = retrieved_knowledge 
                
        prediction = self.predictor(h)
        return prediction, torch.stack(log_probs, dim=1), step + 1

# ==========================================
# 3. EXPERIMENT EXECUTION WITH OGP
# ==========================================
def run_orthogonal_projection_experiment():
    env = ConceptDriftEnvironment()
    model_A = StandardMLP()
    model_B = RecurrentRLReasoner()
    
    opt_A = optim.Adam(model_A.parameters(), lr=0.005)
    opt_B = optim.Adam(model_B.parameters(), lr=0.005)
    mse_loss = nn.MSELoss()
    
    epochs = 400
    acc_A_mode0, acc_A_mode1 = [], []
    acc_B_mode0, acc_B_mode1 = [], []
    epochs_logged = []
    
    # Buffers to capture Phase 1 input spaces
    phase1_inputs = []
    
    print("--- STARTING UNBIASED GRADIENT PROJECTION EXPERIMENT ---")
    
    # Initialize projection matrix container
    I_P = None

    for epoch in range(epochs):
        current_training_mode = 0 if epoch < 200 else 1
        X_train, Y_train = env.generate_batch(batch_size=64, mode=current_training_mode)
        
        if epoch < 200:
            phase1_inputs.append(X_train.detach().clone())
            
        # ----------------------------------------------------
        # CRITICAL TRANSITION: Compute Orthogonal Subspace
        # ----------------------------------------------------
        if epoch == 200:
            print("\n[Concept Drift Triggered] Computing Orthogonal Subspace Matrix...")
            X_0 = torch.cat(phase1_inputs, dim=0)
            R = torch.matmul(X_0.t(), X_0) / X_0.shape[0]
            U, S, V = torch.svd(R)
            
            # Select components explaining 99% of variance
            variance_explained = torch.cumsum(S, dim=0) / torch.sum(S)
            k = torch.where(variance_explained >= 0.99)[0][0].item() + 1
            
            U_k = U[:, :k]
            P = torch.matmul(U_k, U_k.t())
            I_P = torch.eye(5) - P
            print(f"Retained {k}/5 principal dimensions for Orthogonal Protection.\n")

        # --- Train Model A ---
        opt_A.zero_grad()
        pred_A = model_A(X_train)
        loss_A = mse_loss(pred_A, Y_train)
        loss_A.backward()
        
        # Apply OGP to Model A input layer during Phase 2
        if epoch >= 200 and I_P is not None:
            model_A.net[0].weight.grad = torch.matmul(model_A.net[0].weight.grad, I_P)
            
        opt_A.step()
        
        # --- Train Model B ---
        opt_B.zero_grad()
        pred_B, log_probs, steps_taken = model_B.forward_trajectory(X_train)
        pred_loss_B = mse_loss(pred_B, Y_train)
        errors = torch.abs(pred_B - Y_train).detach().squeeze()
        rewards = (1.0 / (1.0 + errors)) - (0.05 * steps_taken)
        rl_loss = -(log_probs.mean(dim=1) * rewards).mean()
        total_loss_B = pred_loss_B + rl_loss
        total_loss_B.backward()
        
        # Apply OGP to Model B input encoder layer during Phase 2
        if epoch >= 200 and I_P is not None:
            model_B.state_encoder.weight.grad = torch.matmul(model_B.state_encoder.weight.grad, I_P)
            
        opt_B.step()
        
        # --- Evaluation Tracking ---
        if epoch % 10 == 0:
            epochs_logged.append(epoch)
            X_test0, Y_test0 = env.generate_batch(100, mode=0)
            X_test1, Y_test1 = env.generate_batch(100, mode=1)
            
            def eval_model(X, Y, model, is_rl=False):
                with torch.no_grad():
                    pred = model.forward_trajectory(X)[0] if is_rl else model(X)
                    return ((torch.abs(pred - Y) < 0.75).sum().item() / len(Y)) * 100

            acc_A_mode0.append(eval_model(X_test0, Y_test0, model_A))
            acc_A_mode1.append(eval_model(X_test1, Y_test1, model_A))
            acc_B_mode0.append(eval_model(X_test0, Y_test0, model_B, is_rl=True))
            acc_B_mode1.append(eval_model(X_test1, Y_test1, model_B, is_rl=True))

    # ==========================================
    # RENDER METRICS
    # ==========================================
    plt.figure(figsize=(12, 7))
    plt.style.use('bmh')
    
    plt.plot(epochs_logged, acc_A_mode0, color='#d62728', linestyle='-', linewidth=2.5, label='Model A (Static MLP) - Mode 0')
    plt.plot(epochs_logged, acc_A_mode1, color='#ff7f0e', linestyle='--', linewidth=2, label='Model A (Static MLP) - Mode 1')
    plt.plot(epochs_logged, acc_B_mode0, color='#1f77b4', linestyle='-', linewidth=2.5, label='Model B (RL+RAG) - Mode 0')
    plt.plot(epochs_logged, acc_B_mode1, color='#2ca02c', linestyle='--', linewidth=2, label='Model B (RL+RAG) - Mode 1')
    
    plt.axvline(x=200, color='black', linestyle='-.', linewidth=2, label='Concept Drift & OGP Injection')
    plt.title('Non-Stationary Evaluation under Orthogonal Gradient Projection', fontsize=14, pad=15)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Fuzzy Accuracy (%)', fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.ylim(0, 105)
    plt.subplots_adjust(right=0.7)
    plt.show()

if __name__ == "__main__":
    run_orthogonal_projection_experiment()