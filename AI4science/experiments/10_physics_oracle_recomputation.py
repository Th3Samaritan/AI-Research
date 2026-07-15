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
# 1. THE PHYSICS ORACLE (DETERMINISTIC FIX)
# ==========================================
class AdvancedPhysicsEnvironment:
    def __init__(self):
        self.formula_maps = {
            0: [0, 1, 2, 3],       # v = u + at
            1: [0, 2, 3, 4],       # s = ut + 0.5at^2
            2: [0, 1, 2, 4],       # v^2 = u^2 + 2as
            3: [2, 5, 6],          # F = ma
            4: [1, 5, 7],          # p = mv
            5: [1, 5, 8]           # KE = 0.5mv^2
        }

    def generate_problem(self, domain_shock=False):
        low, high = (20.0, 50.0) if domain_shock else (1.0, 5.0)
        
        u, a, t, m = np.random.uniform(low, high, 4)
        v = u + (a * t)
        s = (u * t) + (0.5 * a * (t ** 2))
        F = m * a
        p = m * v
        KE = 0.5 * m * (v ** 2)
        
        state = [u, v, a, t, s, m, F, p, KE]
        
        target_formula = random.randint(0, 5)
        target_idx = random.choice(self.formula_maps[target_formula])
        true_val = state[target_idx]
        
        # BUGFIX 1: Mask everything EXCEPT the variables for the chosen formula
        # This makes the structural routing deterministic rather than contradictory.
        input_state = [-100.0] * 9
        for idx in self.formula_maps[target_formula]:
            if idx != target_idx:
                input_state[idx] = state[idx]
        
        state_tensor = torch.tensor(input_state + [target_idx], dtype=torch.float32)
        return state_tensor, true_val, target_formula

# ==========================================
# 2. CONTINUOUS RECOMPUTATION ARCHITECTURE
# ==========================================
class Primitive(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class PrimitiveLibrary(nn.Module):
    def __init__(self, num_primitives, input_dim, output_dim):
        super().__init__()
        self.primitives = nn.ModuleList([
            Primitive(input_dim, output_dim)
            for _ in range(num_primitives)
        ])

class Controller(nn.Module):
    def __init__(self, input_dim, num_primitives):
        super().__init__()
        self.gru = nn.GRU(input_dim, 64, batch_first=True)
        self.selector = nn.Linear(64, num_primitives)

    def forward(self, x_seq):
        _, hidden = self.gru(x_seq)
        logits = self.selector(hidden.squeeze(0))
        return logits, torch.softmax(logits, dim=-1)

class Executor(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.head = nn.Linear(output_dim, 1)

    def forward(self, primitive_output):
        return self.head(primitive_output)

class ReasoningSystem(nn.Module):
    def __init__(self, input_dim=10, num_primitives=6, output_dim=16):
        super().__init__()
        self.library = PrimitiveLibrary(num_primitives, input_dim, output_dim)
        self.controller = Controller(input_dim, num_primitives)
        self.executor = Executor(output_dim)

    def forward(self, x_current):
        x_seq = x_current.unsqueeze(1) 
        logits, selection_weights = self.controller(x_seq)  

        outputs = torch.stack([
            p(x_current) for p in self.library.primitives
        ], dim=1)  

        # BUGFIX 2 & 3: Structural Bulkhead + Hard Routing.
        # 1. Detach selection weights to prevent MSE explosion from frying the Controller.
        # 2. Hard route: Only the selected primitive passes gradients backward, forcing commitment.
        selected = torch.argmax(selection_weights.detach(), dim=-1)
        batch_indices = torch.arange(x_current.size(0), device=x_current.device)
        
        # Pick only the output of the chosen primitive for each item in the batch
        routed_output = outputs[batch_indices, selected]
        
        final_pred = self.executor(routed_output)
        
        return final_pred, selection_weights, logits

# ==========================================
# 3. EXECUTION LOOP & ASYMMETRIC OPTIMIZATION
# ==========================================
def run_recomputation_proof():
    env = AdvancedPhysicsEnvironment()
    model = ReasoningSystem()
    
    controller_optimizer = optim.Adam(model.controller.parameters(), lr=5e-3)
    executor_optimizer = optim.Adam(
        list(model.executor.parameters()) + list(model.library.parameters()), 
        lr=5e-3
    )
    
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    epochs = 400
    epochs_logged = []
    
    ctrl_acc_norm, ctrl_acc_shock = [], []
    exec_acc_norm, exec_acc_shock = [], []
    
    print("--- INITIATING CONTINUOUS RECOMPUTATION PROOF (FIXED) ---")
    
    for epoch in range(epochs):
        is_shock = epoch >= 200
        batch_states, batch_vals, batch_forms = zip(*[env.generate_problem(domain_shock=is_shock) for _ in range(64)])
        
        X = torch.stack(batch_states)
        Y = torch.tensor(batch_vals, dtype=torch.float32).unsqueeze(1)
        Y_form = torch.tensor(batch_forms, dtype=torch.long)
        
        controller_optimizer.zero_grad()
        executor_optimizer.zero_grad()
        
        preds, _, logits = model(X)
        
        # Completely decoupled gradient flow
        loss_exec = mse_loss(preds, Y)
        loss_ctrl = ce_loss(logits, Y_form)
        
        loss_exec.backward()
        loss_ctrl.backward()
        
        controller_optimizer.step()
        executor_optimizer.step()
        
        if epoch % 10 == 0:
            epochs_logged.append(epoch)
            
            def evaluate(shock_flag):
                c_correct, e_correct = 0, 0
                for _ in range(100):
                    t_x, t_y, t_form = env.generate_problem(domain_shock=shock_flag)
                    with torch.no_grad():
                        pred, weights, _ = model(t_x.unsqueeze(0))
                        
                        if torch.argmax(weights).item() == t_form: c_correct += 1
                        
                        tol = 5.0 if shock_flag else 0.5 
                        if abs(pred.item() - t_y) < tol: e_correct += 1
                return c_correct, e_correct

            cn, en = evaluate(False)
            cs, es = evaluate(True)
            
            ctrl_acc_norm.append(cn); exec_acc_norm.append(en)
            ctrl_acc_shock.append(cs); exec_acc_shock.append(es)

    plt.figure(figsize=(12, 7))
    plt.style.use('bmh')
    
    plt.plot(epochs_logged, ctrl_acc_norm, color='#1f77b4', linewidth=2.5, label='Controller Logic (In-Distribution)')
    plt.plot(epochs_logged, ctrl_acc_shock, color='#1f77b4', linestyle='--', linewidth=2.5, label='Controller Logic (DOMAIN SHOCK)')
    
    plt.plot(epochs_logged, exec_acc_norm, color='#d62728', linewidth=2.5, label='Executor Output (In-Distribution)')
    plt.plot(epochs_logged, exec_acc_shock, color='#d62728', linestyle='--', linewidth=2.5, label='Executor Output (DOMAIN SHOCK)')
    
    plt.axvline(x=200, color='black', linestyle='-.', linewidth=2, label='Domain Shock Induced')
    plt.title('Continuous Recomputation: Routing Stability vs Execution Degradation', fontsize=14, pad=15)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(loc='lower left')
    plt.ylim(0, 105)
    plt.show()

if __name__ == "__main__":
    run_recomputation_proof()