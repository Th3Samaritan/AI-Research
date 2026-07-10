import os
# We set this environment variable to resolve the duplicate OpenMP runtime link conflict 
# between PyTorch and Matplotlib's rendering backend on local Windows systems.
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
# 1. THE EXPANDED PHYSICS ORACLE
# ==========================================
class PhysicsEnvironment:
    """
    Simulates a broader physics domain.
    0: v = u + at (Kinematics 1)
    1: s = ut + 0.5at^2 (Kinematics 2)
    2: v^2 = u^2 + 2as (Kinematics 3)
    3: F = ma (Force / Dynamics)
    4: p = mv (Momentum)
    5: K = 0.5mv^2 (Kinetic Energy)
    
    State Vector: [v, u, a, t, s, F, m, p, K] (Size: 9)
    """
    def generate_problem(self, shock_mode=False):
        # DOMAIN SHOCK: 
        # Normal training uses small numbers. Shock testing uses unseen massive numbers.
        low, high = (20.0, 50.0) if shock_mode else (1.0, 5.0)
        
        formula_choice = random.randint(0, 5)
        state = {k: 0.0 for k in ['v', 'u', 'a', 't', 's', 'F', 'm', 'p', 'K']}
        
        if formula_choice == 0:
            u = np.random.uniform(low, high)
            a = np.random.uniform(low, high)
            t = np.random.uniform(low, high)
            subset = {'v': u + a * t, 'u': u, 'a': a, 't': t}
        elif formula_choice == 1:
            u = np.random.uniform(low, high)
            a = np.random.uniform(low, high)
            t = np.random.uniform(low, high)
            subset = {'s': u*t + 0.5*a*(t**2), 'u': u, 'a': a, 't': t}
        elif formula_choice == 2:
            u = np.random.uniform(low, high)
            a = np.random.uniform(low, high)
            s = np.random.uniform(low, high)
            subset = {'v': np.sqrt(u**2 + 2*a*s), 'u': u, 'a': a, 's': s}
        elif formula_choice == 3:
            m = np.random.uniform(low, high)
            a = np.random.uniform(low, high)
            subset = {'F': m * a, 'm': m, 'a': a}
        elif formula_choice == 4:
            m = np.random.uniform(low, high)
            v = np.random.uniform(low, high)
            subset = {'p': m * v, 'm': m, 'v': v}
        elif formula_choice == 5:
            m = np.random.uniform(low, high)
            v = np.random.uniform(low, high)
            subset = {'K': 0.5 * m * (v**2), 'm': m, 'v': v}
            
        state.update(subset)
        
        # Randomly select one variable from the active subset to be the "Unknown"
        target_key = random.choice(list(subset.keys()))
        true_val = state[target_key]
        state[target_key] = -100.0 # Mask
        
        order = ['v', 'u', 'a', 't', 's', 'F', 'm', 'p', 'K']
        state_tensor = torch.tensor([state[k] for k in order], dtype=torch.float32)
        return state_tensor, target_key, true_val

# ==========================================
# 2. PARADIGM A: STATIC BIG-DATA MODEL
# ==========================================
class BigDataPatternMatcher(nn.Module):
    """
    Expanded capacity Multi-Layer Perceptron.
    Takes 9 inputs and tries to map 6 different non-linear equations entirely via MSE loss.
    """
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
    """
    Reasoning Agent. Input size 9, Action space 6 (selects the law of physics).
    """
    def __init__(self):
        super(AutonomousRLAgent, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6) # Probabilities for the 6 formulas
        )
        
    def forward(self, state):
        logits = self.policy_net(state)
        return torch.softmax(logits, dim=-1)

    def execute_rag_and_solve(self, state, formula_idx):
        v, u, a, t, s, F, m, p, K = [val.item() for val in state]
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
                if t == -100.0: return (-u + np.sqrt(u**2 + 2*a*s)) / a
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
            elif formula_idx == 5: # K = 0.5mv^2
                if K == -100.0: return 0.5 * m * (v**2)
                if m == -100.0: return (2 * K) / (v**2)
                if v == -100.0: return np.sqrt(abs((2 * K) / m))
            return -9999.0
        except Exception:
            return -9999.0

# ==========================================
# 4. THE EXPERIMENT EXECUTION LOOP
# ==========================================
def run_local_experiment():
    env = PhysicsEnvironment()
    
    print("[1/3] Generating 10,000 Static IN-DISTRIBUTION samples for Paradigm A...")
    static_data = [env.generate_problem(shock_mode=False) for _ in range(10000)]
    
    model_A = BigDataPatternMatcher()
    optimizer_A = optim.Adam(model_A.parameters(), lr=0.001)
    loss_fn_A = nn.MSELoss()
    
    model_B = AutonomousRLAgent()
    optimizer_B = optim.Adam(model_B.parameters(), lr=0.01)
    
    epochs = 400
    epochs_logged = []
    
    # Track 4 distinct histories for the Proof
    hist_A_ID = []
    hist_A_OOD = []
    hist_B_ID = []
    hist_B_OOD = []
    
    print("[2/3] Starting Training Loops (400 Epochs)...")
    print("-" * 80)
    
    for epoch in range(epochs):
        # ---------------------------------------------------------
        # Train Model A (Supervised mapping on ID Data)
        # ---------------------------------------------------------
        batch_A = random.sample(static_data, 64)
        states_A = torch.stack([b[0] for b in batch_A])
        targets_A = torch.tensor([[b[2]] for b in batch_A], dtype=torch.float32)
        
        optimizer_A.zero_grad()
        predictions_A = model_A(states_A)
        loss_A = loss_fn_A(predictions_A, targets_A)
        loss_A.backward()
        optimizer_A.step()
        
        # ---------------------------------------------------------
        # Train Model B (Continuous Thought via RL Feedback on ID Data)
        # ---------------------------------------------------------
        optimizer_B.zero_grad()
        rl_loss = 0
        
        for _ in range(64):
            state_B, _, true_val_B = env.generate_problem(shock_mode=False)
            
            probs = model_B(state_B)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            agent_answer = model_B.execute_rag_and_solve(state_B, action.item())
            
            if abs(agent_answer - true_val_B) < 0.1:
                reward = 1.0
            else:
                reward = -1.0
                
            rl_loss += -m.log_prob(action) * reward
            
        rl_loss.backward()
        optimizer_B.step()
        
        # ---------------------------------------------------------
        # Evaluation: Testing In-Distribution (ID) vs Domain Shock (OOD)
        # ---------------------------------------------------------
        if epoch % 20 == 0 or epoch == epochs - 1:
            cA_ID, cA_OOD, cB_ID, cB_OOD = 0, 0, 0, 0
            
            for _ in range(100):
                # 1. Test In-Distribution (Familiar variable ranges)
                test_s_ID, _, test_v_ID = env.generate_problem(shock_mode=False)
                
                if abs(model_A(test_s_ID.unsqueeze(0)).item() - test_v_ID) < 0.5: cA_ID += 1
                if abs(model_B.execute_rag_and_solve(test_s_ID, torch.argmax(model_B(test_s_ID)).item()) - test_v_ID) < 0.1: cB_ID += 1
                
                # 2. Test Out-Of-Distribution (Massive variables to break pattern matchers)
                test_s_OOD, _, test_v_OOD = env.generate_problem(shock_mode=True)
                
                if abs(model_A(test_s_OOD.unsqueeze(0)).item() - test_v_OOD) < 0.5: cA_OOD += 1
                if abs(model_B.execute_rag_and_solve(test_s_OOD, torch.argmax(model_B(test_s_OOD)).item()) - test_v_OOD) < 0.1: cB_OOD += 1

            hist_A_ID.append(cA_ID)
            hist_A_OOD.append(cA_OOD)
            hist_B_ID.append(cB_ID)
            hist_B_OOD.append(cB_OOD)
            epochs_logged.append(epoch)
            
            print(f"Epoch {epoch:03d} | A (ID): {cA_ID}% | A (OOD): {cA_OOD}% || B (ID): {cB_ID}% | B (OOD): {cB_OOD}%")

    # ==========================================
    # 5. GRAPH PLOTTING
    # ==========================================
    print("-" * 80)
    print("[3/3] Training complete. Rendering Proof...")
    
    plt.figure(figsize=(12, 7))
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.plot(epochs_logged, hist_A_ID, label='Paradigm A (Big-Data) - Familiar Variables', color='#d62728', linewidth=2.5, linestyle='-')
    plt.plot(epochs_logged, hist_A_OOD, label='Paradigm A (Big-Data) - DOMAIN SHOCK', color='#ff7f0e', linewidth=2.5, linestyle='--')
    
    plt.plot(epochs_logged, hist_B_ID, label='Paradigm B (RL Reasoner) - Familiar Variables', color='#1f77b4', linewidth=2.5, linestyle='-')
    plt.plot(epochs_logged, hist_B_OOD, label='Paradigm B (RL Reasoner) - DOMAIN SHOCK', color='#2ca02c', linewidth=2.5, linestyle='--')
    
    plt.title('Research Proof: Memorization vs. Universal Reasoning', fontsize=16, pad=15)
    plt.xlabel('Training Generations (Epochs)', fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)
    
    # Adding a clean legend
    plt.legend(loc='center right', fontsize=11, framealpha=0.9)
    plt.ylim(-5, 105)
    plt.tight_layout()
    
    output_image = "domain_shock_proof.png"
    plt.savefig(output_image, dpi=300)
    print(f"Graph saved as '{output_image}'. Close the pop-up to exit.")
    
    plt.show()

if __name__ == "__main__":
    run_local_experiment()