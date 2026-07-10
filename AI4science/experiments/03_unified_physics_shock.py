import os
# Override to prevent local Windows/OpenMP thread crashes with PyTorch and Matplotlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION & UNBIASED SEEDING
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==========================================
# 1. THE UNIFIED PHYSICS ORACLE
# ==========================================
class AdvancedPhysicsEnvironment:
    """
    Simulates a unified physics state with 9 variables.
    Indices: 0:u, 1:v, 2:a, 3:t, 4:s, 5:m, 6:F, 7:p, 8:KE
    """
    def __init__(self):
        # Maps formula index to the variable indices it utilizes
        self.formula_maps = {
            0: [0, 1, 2, 3],       # v = u + at
            1: [0, 2, 3, 4],       # s = ut + 0.5at^2
            2: [0, 1, 2, 4],       # v^2 = u^2 + 2as
            3: [2, 5, 6],          # F = ma
            4: [1, 5, 7],          # p = mv
            5: [1, 5, 8]           # KE = 0.5mv^2
        }

    def generate_problem(self, domain_shock=False):
        # Domain Shock: completely out-of-distribution numbers
        low, high = (20.0, 50.0) if domain_shock else (1.0, 5.0)
        
        # Base independent variables
        u = np.random.uniform(low, high)
        a = np.random.uniform(low, high)
        t = np.random.uniform(low, high)
        m = np.random.uniform(low, high)
        
        # Derived dependent variables
        v = u + (a * t)
        s = (u * t) + (0.5 * a * (t ** 2))
        F = m * a
        p = m * v
        KE = 0.5 * m * (v ** 2)
        
        state = [u, v, a, t, s, m, F, p, KE]
        
        # Guarantee 1-step solvability by masking a variable from a specific valid formula
        target_formula = random.randint(0, 5)
        target_idx = random.choice(self.formula_maps[target_formula])
        
        true_val = state[target_idx]
        
        input_state = list(state)
        input_state[target_idx] = -100.0 # Mask the unknown
        
        state_tensor = torch.tensor(input_state, dtype=torch.float32)
        return state_tensor, target_idx, true_val

# ==========================================
# 2. PARADIGM A: STATIC BIG-DATA MODEL
# ==========================================
class BigDataPatternMatcher(nn.Module):
    def __init__(self):
        super(BigDataPatternMatcher, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. PARADIGM B: CONTINUOUS THOUGHT + RL AGENT
# ==========================================
class AutonomousRLAgent(nn.Module):
    def __init__(self):
        super(AutonomousRLAgent, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6) # Now predicts across 6 distinct physics laws
        )
        
    def forward(self, state):
        logits = self.policy_net(state)
        return torch.softmax(logits, dim=-1)

    def execute_rag_and_solve(self, state, formula_idx):
        # Extracts the 9 variables
        u, v, a, t, s, m, F, p, KE = [state[i].item() for i in range(9)]
        
        try:
            if formula_idx == 0: # v = u + at
                if v == -100.0: return u + (a * t)
                if u == -100.0: return v - (a * t)
                if a == -100.0: return (v - u) / t
                if t == -100.0: return (v - u) / a
            elif formula_idx == 1: # s = ut + 0.5at^2
                if s == -100.0: return (u * t) + (0.5 * a * (t**2))
                if u == -100.0: return (s - (0.5 * a * (t**2))) / t
                if a == -100.0: return (2 * (s - (u * t))) / (t**2)
            elif formula_idx == 2: # v^2 = u^2 + 2as
                if v == -100.0: return np.sqrt(abs((u**2) + (2 * a * s)))
                if u == -100.0: return np.sqrt(abs((v**2) - (2 * a * s)))
                if a == -100.0: return ((v**2) - (u**2)) / (2 * s)
                if s == -100.0: return ((v**2) - (u**2)) / (2 * a)
            elif formula_idx == 3: # F = ma
                if F == -100.0: return m * a
                if m == -100.0: return F / a
                if a == -100.0: return F / m
            elif formula_idx == 4: # p = mv
                if p == -100.0: return m * v
                if m == -100.0: return p / v
                if v == -100.0: return p / m
            elif formula_idx == 5: # KE = 0.5mv^2
                if KE == -100.0: return 0.5 * m * (v**2)
                if m == -100.0: return (2 * KE) / (v**2)
                if v == -100.0: return np.sqrt(abs((2 * KE) / m))
            return -9999.0 # Picked wrong formula
        except Exception:
            return -9999.0

# ==========================================
# 4. EXPERIMENT EXECUTION LOOP
# ==========================================
def run_research_proof():
    env = AdvancedPhysicsEnvironment()
    
    print("[1/3] Generating 10,000 static corpus samples (In-Distribution)...")
    static_data = [env.generate_problem(domain_shock=False) for _ in range(10000)]
    
    model_A = BigDataPatternMatcher()
    optimizer_A = optim.Adam(model_A.parameters(), lr=0.001)
    loss_fn_A = nn.MSELoss()
    
    model_B = AutonomousRLAgent()
    optimizer_B = optim.Adam(model_B.parameters(), lr=0.01)
    
    epochs = 400
    epochs_logged = []
    
    hist_A_normal, hist_A_shock = [], []
    hist_B_normal, hist_B_shock = [], []
    
    print("[2/3] Starting parallel training & verification loops...")
    print("-" * 80)
    
    for epoch in range(epochs):
        # -- Train Paradigm A (Supervised) --
        batch_A = random.sample(static_data, 64)
        states_A = torch.stack([b[0] for b in batch_A])
        targets_A = torch.tensor([[b[2]] for b in batch_A], dtype=torch.float32)
        
        optimizer_A.zero_grad()
        loss_A = loss_fn_A(model_A(states_A), targets_A)
        loss_A.backward()
        optimizer_A.step()
        
        # -- Train Paradigm B (RL Continuous Thought) --
        optimizer_B.zero_grad()
        rl_loss = 0
        
        for _ in range(64):
            # Only trains on normal distribution
            state_B, _, true_val_B = env.generate_problem(domain_shock=False)
            
            probs = model_B(state_B)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            agent_answer = model_B.execute_rag_and_solve(state_B, action.item())
            
            # Reward formulation based on mathematical truth
            reward = 1.0 if abs(agent_answer - true_val_B) < 0.1 else -1.0
            rl_loss += -m.log_prob(action) * reward
            
        rl_loss.backward()
        optimizer_B.step()
        
        # -- Evaluation (Testing Generalization) --
        if epoch % 20 == 0 or epoch == epochs - 1:
            cA_norm, cA_shock = 0, 0
            cB_norm, cB_shock = 0, 0
            
            for _ in range(100):
                # 1. Test In-Distribution
                s_norm, _, v_norm = env.generate_problem(domain_shock=False)
                if abs(model_A(s_norm.unsqueeze(0)).item() - v_norm) < 0.5: cA_norm += 1
                act_B_norm = torch.argmax(model_B(s_norm)).item()
                if abs(model_B.execute_rag_and_solve(s_norm, act_B_norm) - v_norm) < 0.1: cB_norm += 1
                
                # 2. Test Out-of-Distribution (The Shock)
                s_shock, _, v_shock = env.generate_problem(domain_shock=True)
                # Model A requires a massively generous tolerance here just to survive
                if abs(model_A(s_shock.unsqueeze(0)).item() - v_shock) < 5.0: cA_shock += 1
                act_B_shock = torch.argmax(model_B(s_shock)).item()
                if abs(model_B.execute_rag_and_solve(s_shock, act_B_shock) - v_shock) < 0.1: cB_shock += 1

            hist_A_normal.append(cA_norm)
            hist_A_shock.append(cA_shock)
            hist_B_normal.append(cB_norm)
            hist_B_shock.append(cB_shock)
            epochs_logged.append(epoch)
            
            print(f"Epoch {epoch:03d} | Model A (Normal/Shock): {cA_norm}% / {cA_shock}% | Model B (Normal/Shock): {cB_norm}% / {cB_shock}%")

    # ==========================================
    # 5. RENDER THE RESEARCH GRAPH
    # ==========================================
    print("-" * 80)
    print("[3/3] Training complete. Rendering Out-of-Distribution proof...")
    
    plt.figure(figsize=(11, 7))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # In-Distribution Lines (Dashed)
    plt.plot(epochs_logged, hist_A_normal, color='#d62728', linestyle='--', alpha=0.5,
             label='Paradigm A: Big-Data (In-Distribution)')
    plt.plot(epochs_logged, hist_B_normal, color='#1f77b4', linestyle='--', alpha=0.5,
             label='Paradigm B: RL Reasoner (In-Distribution)')
             
    # Out-of-Distribution Shock Lines (Solid, Thick)
    plt.plot(epochs_logged, hist_A_shock, color='#8c564b', linewidth=3,
             label='Paradigm A: Big-Data (DOMAIN SHOCK)')
    plt.plot(epochs_logged, hist_B_shock, color='#2ca02c', linewidth=3,
             label='Paradigm B: RL Reasoner (DOMAIN SHOCK)')
    
    plt.title('Out-of-Distribution Generalization:\nPattern Matching vs. Autonomous Reasoning', fontsize=15, pad=15)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(loc='center right', fontsize=10)
    plt.ylim(-5, 105)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_research_proof()