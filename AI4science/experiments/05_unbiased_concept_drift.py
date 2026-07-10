import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==========================================
# 1. NON-STATIONARY FUZZY ENVIRONMENT
# ==========================================
class ConceptDriftEnvironment:
    """
    Generates fuzzy, noisy continuous data.
    The underlying mathematical rules change based on the 'mode'.
    """
    def generate_batch(self, batch_size=32, mode=0):
        # x0 to x3 are core variables, x4 is a subtle contextual cue (sensor signature)
        X = torch.rand(batch_size, 5) * 4.0 
        
        # Inject context signature with heavy noise
        if mode == 0:
            X[:, 4] = torch.normal(mean=-2.0, std=0.5, size=(batch_size,))
            # Rule 0: Non-linear interaction A
            Y = (X[:, 0] * X[:, 1]) + torch.sin(X[:, 2]) - X[:, 3]
        else:
            X[:, 4] = torch.normal(mean=2.0, std=0.5, size=(batch_size,))
            # Rule 1: Completely different non-linear interaction B
            Y = (X[:, 2] ** 2) - torch.cos(X[:, 0] * X[:, 1]) + X[:, 3]
            
        # Add fuzzy environmental noise to the final target
        Y += torch.normal(mean=0.0, std=0.2, size=(batch_size,))
        return X, Y.unsqueeze(1)

# ==========================================
# 2. PARADIGM A: PURE NEURAL MLP
# ==========================================
class StandardMLP(nn.Module):
    """
    A standard Deep Neural Network.
    Must learn to map X to Y implicitly through its weights.
    """
    def __init__(self):
        super(StandardMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. PARADIGM B: CONTINUOUS THOUGHT (RL + RAG)
# ==========================================
class RecurrentRLReasoner(nn.Module):
    """
    An agent that thinks sequentially.
    Actions: 0: Query RAG Mode 0, 1: Query RAG Mode 1, 2: Stop & Predict
    """
    def __init__(self):
        super(RecurrentRLReasoner, self).__init__()
        
        # RAG Database (Learnable Embeddings representing 'Theories')
        self.rag_memory = nn.Embedding(2, 16) 
        
        # Continuous Thought Engine
        self.state_encoder = nn.Linear(5, 16)
        self.gru = nn.GRUCell(input_size=16, hidden_size=32)
        
        # RL Policy Head (Actor)
        self.policy = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3) # Probabilities for the 3 actions
        )
        
        # Prediction Head (Neural mapping, NO perfect math calculator)
        self.predictor = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward_trajectory(self, x, max_steps=3):
        batch_size = x.size(0)
        
        # Initialize thought state
        encoded_x = self.state_encoder(x)
        h = torch.zeros(batch_size, 32)
        
        log_probs = []
        
        # Continuous Thought Loop
        for step in range(max_steps):
            h = self.gru(encoded_x, h)
            
            # Policy decides action based on current thought
            action_logits = self.policy(h)
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            log_probs.append(action_dist.log_prob(action))
            
            # For simplicity in batch processing, we force a homogeneous action path
            # In a true asynchronous setup, each batch item stops independently.
            majority_action = torch.mode(action).values.item()
            
            if majority_action == 2: # STOP and PREDICT
                break
                
            elif majority_action in [0, 1]: # QUERY RAG
                # Retrieve knowledge and feed it back into the thought loop
                retrieved_knowledge = self.rag_memory(torch.tensor([majority_action] * batch_size))
                # Next input to GRU is the retrieved knowledge
                encoded_x = retrieved_knowledge 
                
        prediction = self.predictor(h)
        return prediction, torch.stack(log_probs, dim=1), step + 1

# ==========================================
# 4. EXPERIMENT EXECUTION
# ==========================================
def run_unbiased_concept_drift_proof():
    env = ConceptDriftEnvironment()
    
    model_A = StandardMLP()
    model_B = RecurrentRLReasoner()
    
    # We use shared optimizers for prediction (MSE) and policy (RL)
    opt_A = optim.Adam(model_A.parameters(), lr=0.005)
    opt_B = optim.Adam(model_B.parameters(), lr=0.005)
    mse_loss = nn.MSELoss()
    
    epochs = 400
    
    # Tracking accuracy on BOTH modes simultaneously
    acc_A_mode0, acc_A_mode1 = [], []
    acc_B_mode0, acc_B_mode1 = [], []
    epochs_logged = []
    
    print("--- STARTING UNBIASED CONTINUOUS LEARNING PROOF ---")
    print("Epochs 0-200  : Training heavily on Mode 0 (Environment State A)")
    print("Epochs 200-400: Silent Concept Drift -> Training heavily on Mode 1 (Environment State B)")
    print("-" * 70)
    
    for epoch in range(epochs):
        # ----------------------------------------------------
        # THE CONCEPT DRIFT: Shift the training distribution
        # ----------------------------------------------------
        current_training_mode = 0 if epoch < 200 else 1
        X_train, Y_train = env.generate_batch(batch_size=64, mode=current_training_mode)
        
        # -------------------
        # Train Model A (Supervised)
        # -------------------
        opt_A.zero_grad()
        pred_A = model_A(X_train)
        loss_A = mse_loss(pred_A, Y_train)
        loss_A.backward()
        opt_A.step()
        
        # -------------------
        # Train Model B (Supervised + RL)
        # -------------------
        opt_B.zero_grad()
        pred_B, log_probs, steps_taken = model_B.forward_trajectory(X_train)
        
        # Supervised Loss for the neural prediction head
        pred_loss_B = mse_loss(pred_B, Y_train)
        
        # RL Reward formulation for the policy head
        # Positive reward for low error, negative penalty (-0.05) for thinking too long
        errors = torch.abs(pred_B - Y_train).detach().squeeze()
        rewards = (1.0 / (1.0 + errors)) - (0.05 * steps_taken)
        
        # Policy Gradient Loss (REINFORCE)
        rl_loss = -(log_probs.mean(dim=1) * rewards).mean()
        
        # Total Loss is Fusion of accuracy and efficient reasoning
        total_loss_B = pred_loss_B + rl_loss
        total_loss_B.backward()
        opt_B.step()
        
        # ----------------------------------------------------
        # EVALUATION: Test on BOTH modes to check for Forgetting
        # ----------------------------------------------------
        if epoch % 10 == 0:
            epochs_logged.append(epoch)
            
            def eval_model(X_test, Y_test, model, is_rl=False):
                with torch.no_grad():
                    if is_rl:
                        pred, _, _ = model.forward_trajectory(X_test)
                    else:
                        pred = model(X_test)
                    # Tolerance of 0.75 for "Correct" due to fuzzy noise
                    correct = (torch.abs(pred - Y_test) < 0.75).sum().item()
                    return (correct / len(Y_test)) * 100

            # Generate static test sets
            X_test0, Y_test0 = env.generate_batch(100, mode=0)
            X_test1, Y_test1 = env.generate_batch(100, mode=1)
            
            # Evaluate A
            acc_A_mode0.append(eval_model(X_test0, Y_test0, model_A))
            acc_A_mode1.append(eval_model(X_test1, Y_test1, model_A))
            
            # Evaluate B
            acc_B_mode0.append(eval_model(X_test0, Y_test0, model_B, is_rl=True))
            acc_B_mode1.append(eval_model(X_test1, Y_test1, model_B, is_rl=True))

    # ==========================================
    # RENDER THE SCIENTIFIC METRICS
    # ==========================================
    print("Training Complete. Rendering Catastrophic Forgetting Analysis...")
    
    plt.figure(figsize=(12, 7))
    plt.style.use('bmh')
    
    # Model A (Big Data)
    plt.plot(epochs_logged, acc_A_mode0, color='#d62728', linestyle='-', linewidth=2.5, label='Model A (Static MLP) - Mode 0 Acc')
    plt.plot(epochs_logged, acc_A_mode1, color='#ff7f0e', linestyle='--', linewidth=2, label='Model A (Static MLP) - Mode 1 Acc')
    
    # Model B (RL+RAG)
    plt.plot(epochs_logged, acc_B_mode0, color='#1f77b4', linestyle='-', linewidth=2.5, label='Model B (RL+RAG Reasoner) - Mode 0 Acc')
    plt.plot(epochs_logged, acc_B_mode1, color='#2ca02c', linestyle='--', linewidth=2, label='Model B (RL+RAG Reasoner) - Mode 1 Acc')
    
    # Concept Drift Indicator
    plt.axvline(x=200, color='black', linestyle='-.', linewidth=2, label='Concept Drift (Environment Shifts)')
    
    plt.title('Catastrophic Forgetting Analysis: Static Models vs. RL/RAG Reasoners', fontsize=14, pad=15)
    plt.xlabel('Training Epochs (Continuous)', fontsize=12)
    plt.ylabel('Fuzzy Accuracy (%)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.45, 1))
    plt.ylim(0, 105)
    plt.subplots_adjust(right=0.7)
    plt.grid(True, alpha=0.6)
    
    plt.show()

if __name__ == "__main__":
    run_unbiased_concept_drift_proof()