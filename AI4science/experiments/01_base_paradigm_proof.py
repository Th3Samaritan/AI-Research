import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os
# ==========================================
# CONFIGURATION & UNBIASED SEEDING
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ==========================================
# 1. THE PHYSICS ORACLE (GROUND TRUTH)
# ==========================================
class PhysicsEnvironment:
    """
    Simulates 1D Kinematics under constant acceleration.
    Formula 0: v = u + at
    Formula 1: s = ut + 0.5at^2
    Formula 2: v^2 = u^2 + 2as
    """
    def generate_problem(self):
        t = np.random.uniform(1.0, 5.0)
        a = np.random.uniform(1.0, 5.0)
        u = np.random.uniform(0.0, 10.0)
        
        v = u + (a * t)
        s = (u * t) + (0.5 * a * (t ** 2))
        
        state_dict = {'u': u, 'v': v, 'a': a, 't': t, 's': s}
        keys = list(state_dict.keys())
        
        # Randomly select one variable to be masked as the unknown (-100.0)
        target_idx = random.randint(0, 4)
        target_key = keys[target_idx]
        true_val = state_dict[target_key]
        
        input_state = state_dict.copy()
        input_state[target_key] = -100.0 
        
        state_tensor = torch.tensor([input_state[k] for k in keys], dtype=torch.float32)
        return state_tensor, target_idx, true_val

# ==========================================
# 2. PARADIGM A: STATIC BIG-DATA MODEL
# ==========================================
class BigDataPatternMatcher(nn.Module):
    """
    Standard Multi-Layer Perceptron representing supervised pattern recognition.
    Trained on 10,000 static examples via Mean Squared Error (MSE) loss.
    """
    def __init__(self):
        super(BigDataPatternMatcher, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
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
    Small-Data reasoning agent. It evaluates the environment and decides
    which fundamental physics law from its RAG memory to deploy.
    """
    def __init__(self):
        super(AutonomousRLAgent, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Selection weight for Formulas 0, 1, or 2
        )
        
    def forward(self, state):
        logits = self.policy_net(state)
        return torch.softmax(logits, dim=-1)

    def execute_rag_and_solve(self, state, formula_idx):
        u, v, a, t, s = state[0].item(), state[1].item(), state[2].item(), state[3].item(), state[4].item()
        
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
            return -9999.0
        except Exception:
            return -9999.0

# ==========================================
# 4. THE EXPERIMENT EXECUTION LOOP
# ==========================================
def run_local_experiment():
    env = PhysicsEnvironment()
    
    print("[1/3] Generating 10,000 static corpus samples for Paradigm A...")
    static_data = [env.generate_problem() for _ in range(10000)]
    
    model_A = BigDataPatternMatcher()
    optimizer_A = optim.Adam(model_A.parameters(), lr=0.001)
    loss_fn_A = nn.MSELoss()
    
    model_B = AutonomousRLAgent()
    optimizer_B = optim.Adam(model_B.parameters(), lr=0.01)
    
    epochs = 400
    epochs_logged = []  # Explicitly tracking x-axis steps to avoid shape errors
    history_A = []
    history_B = []
    
    print("[2/3] Starting parallel training loops (400 Epochs)...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # ---------------------------------------------------------
        # Train Model A (Pattern Matching on Massive Dataset)
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
        # Train Model B (Continuous Thought via RL Feedback)
        # ---------------------------------------------------------
        optimizer_B.zero_grad()
        rl_loss = 0
        
        for _ in range(64):
            state_B, _, true_val_B = env.generate_problem()
            
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
        # Evaluation & Console Logs (Every 20 Epochs)
        # ---------------------------------------------------------
        if epoch % 20 == 0 or epoch == epochs - 1:
            correct_A, correct_B = 0, 0
            
            for _ in range(100):
                test_s, _, test_v = env.generate_problem()
                
                # Evaluate Paradigm A
                pred_a = model_A(test_s.unsqueeze(0)).item()
                if abs(pred_a - test_v) < 0.5: 
                    correct_A += 1
                    
                # Evaluate Paradigm B
                probs_b = model_B(test_s)
                action_b = torch.argmax(probs_b).item()
                ans_b = model_B.execute_rag_and_solve(test_s, action_b)
                if abs(ans_b - test_v) < 0.1: 
                    correct_B += 1
            
            history_A.append(correct_A)
            history_B.append(correct_B)
            epochs_logged.append(epoch)  # Record the precise matching epoch number
            
            print(f"Epoch {epoch:03d} | Paradigm A (Big Data) Acc: {correct_A}% | Paradigm B (RL Reasoner) Acc: {correct_B}%")

    # ==========================================
    # 5. GRAPH PLOTTING (LOCAL WINDOW)
    # ==========================================
    print("-" * 70)
    print("[3/3] Training complete. Rendering performance graph window...")
    
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plotting using the dynamically synced x-axis
    plt.plot(epochs_logged, history_A, 
             label='Paradigm A: Big-Data (Supervised Memorization)', color='#d62728', linewidth=2.5, linestyle='--')
    plt.plot(epochs_logged, history_B, 
             label='Paradigm B: Small-Data (Continuous Thought + RL)', color='#1f77b4', linewidth=2.5)
    
    plt.title('AI Paradigm Proof: Memorization vs. Autonomous Reasoning', fontsize=14, pad=15)
    plt.xlabel('Training Generations (Epochs)', fontsize=12)
    plt.ylabel('Accuracy on Unseen Problems (%)', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.ylim(-5, 105)
    
    plt.tight_layout()
    print("Close the graph window to exit the script cleanly.")
    plt.show()

if __name__ == "__main__":
    run_local_experiment()