"""
GOOGLE COLAB INSTRUCTIONS (From Scratch):
1. Runtime -> Change runtime type -> T4 GPU.
2. Run this in a cell first: !pip install -q transformers datasets matplotlib
3. Run this script.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# ==========================================
# 1. DATA PREPARATION
# ==========================================
print("Loading Tokenizer & Datasets...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds_story = load_dataset("roneneldan/TinyStories", split="train[:5000]")
ds_web = load_dataset("Skylion007/openwebtext", split="train[:5000]")

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32, return_tensors="pt")

story_iter = DataLoader(ds_story.map(tokenize_batch, batched=True, remove_columns=ds_story.column_names).with_format("torch"), batch_size=32, shuffle=True)
web_iter = DataLoader(ds_web.map(tokenize_batch, batched=True, remove_columns=ds_web.column_names).with_format("torch"), batch_size=32, shuffle=True)

def cycle(iterable):
    while True:
        for x in iterable: yield x

iter_s = cycle(story_iter)
iter_w = cycle(web_iter)

# ==========================================
# 2. FROM-SCRATCH TRANSFORMER & HARD ROUTING
# ==========================================
class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_head = 4
        self.d_model = d_model

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = SimpleSelfAttention(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ScratchRecomputationLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(32, d_model) # Max length 32
        
        # 1. SHARED CORTEX (Learns basic grammar/syntax)
        self.shared_blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(2)])
        
        # 2. CONTROLLER (The Strategist)
        self.controller = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # [Story, Web]
        )
        
        # 3. PRIMITIVE LIBRARY (The Execution Tools)
        self.primitive_story = TransformerBlock(d_model)
        self.primitive_web = TransformerBlock(d_model)
        
        # 4. EXECUTOR HEAD
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.to(self.device)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=self.device)
        
        # Pass through Shared Cortex
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.shared_blocks:
            x = block(x)
            
        # Controller Routing Phase
        # Look at the sequence mean to determine the context
        pooled_x = x.mean(dim=1)
        routing_logits = self.controller(pooled_x)
        
        # STRUCTURAL BULKHEAD & HARD ROUTING
        # 1. Detach to protect controller from executor gradients
        # 2. Use argmax to perform per-sample routing
        selections = torch.argmax(routing_logits.detach(), dim=-1) # (B,)
        
        routed_x = torch.zeros_like(x)
        story_mask = (selections == 0)
        web_mask = (selections == 1)
        
        # Only selected primitives execute and receive gradient graphs
        if story_mask.any():
            routed_x[story_mask] = self.primitive_story(x[story_mask])
        if web_mask.any():
            routed_x[web_mask] = self.primitive_web(x[web_mask])
            
        # Execution Phase
        x = self.ln_f(routed_x)
        logits = self.lm_head(x)
        
        return logits, routing_logits

# ==========================================
# 3. ASYMMETRIC TRAINING LOOP
# ==========================================
def run_scratch_proof():
    model = ScratchRecomputationLLM(tokenizer.vocab_size)
    
    # Decoupled Optimizers
    controller_optimizer = optim.Adam(model.controller.parameters(), lr=1e-3)
    executor_params = [p for n, p in model.named_parameters() if "controller" not in n]
    executor_optimizer = optim.Adam(executor_params, lr=5e-4)
    
    ce_loss_fn = nn.CrossEntropyLoss()
    steps = 400
    steps_logged, ctrl_story_acc, ctrl_web_acc, exec_story_loss, exec_web_loss = [], [], [], [], []
    
    print("\n--- INITIATING NANO-MoE RECOMPUTATION PROOF ---")
    
    for step in range(steps):
        is_shock = step >= 200
        batch = next(iter_w) if is_shock else next(iter_s)
        true_domain = 1 if is_shock else 0
            
        input_ids = batch["input_ids"].to(model.device)
        domain_labels = torch.full((input_ids.size(0),), true_domain, dtype=torch.long, device=model.device)
        
        # Shift inputs for causal LM (predict next token)
        x_inputs = input_ids[:, :-1]
        y_targets = input_ids[:, 1:].contiguous().view(-1)
        
        controller_optimizer.zero_grad()
        executor_optimizer.zero_grad()
        
        # Forward Pass
        lm_logits, routing_logits = model(x_inputs)
        
        # Decoupled Losses
        lm_loss = ce_loss_fn(lm_logits.view(-1, lm_logits.size(-1)), y_targets)
        ctrl_loss = ce_loss_fn(routing_logits, domain_labels)
        
        # Structural Bulkhead (Retain graph for shared cortex)
        ctrl_loss.backward(retain_graph=True)
        lm_loss.backward()
        
        controller_optimizer.step()
        executor_optimizer.step()
        
        if step % 10 == 0:
            steps_logged.append(step)
            def evaluate(eval_iter, domain_id):
                model.eval()
                with torch.no_grad():
                    eval_batch = next(eval_iter)
                    ids = eval_batch["input_ids"].to(model.device)
                    x_in, y_tgt = ids[:, :-1], ids[:, 1:].contiguous().view(-1)
                    l_logits, r_logits = model(x_in)
                    loss = ce_loss_fn(l_logits.view(-1, l_logits.size(-1)), y_tgt)
                    acc = (torch.argmax(r_logits, dim=-1) == domain_id).float().mean().item() * 100
                model.train()
                return acc, loss.item()
            
            s_acc, s_loss = evaluate(iter_s, 0)
            w_acc, w_loss = evaluate(iter_w, 1)
            ctrl_story_acc.append(s_acc); ctrl_web_acc.append(w_acc)
            exec_story_loss.append(s_loss); exec_web_loss.append(w_loss)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.style.use('bmh')
    
    ax1.plot(steps_logged, ctrl_story_acc, color='#1f77b4', linewidth=2.5, label='Controller: Narrative Context')
    ax1.plot(steps_logged, ctrl_web_acc, color='#ff7f0e', linestyle='--', linewidth=2.5, label='Controller: Informational Context')
    ax1.axvline(x=200, color='black', linestyle='-.', linewidth=2, label='Domain Shock Induced')
    ax1.set_title('Controller Routing Stability (True Hardware Routing)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend(loc='lower left')
    ax1.set_ylim(-5, 105)
    
    ax2.plot(steps_logged, exec_story_loss, color='#1f77b4', linewidth=2.5, label='Executor: Story LM Loss')
    ax2.plot(steps_logged, exec_web_loss, color='#ff7f0e', linestyle='--', linewidth=2.5, label='Executor: Web LM Loss')
    ax2.axvline(x=200, color='black', linestyle='-.', linewidth=2, label='Domain Shock Induced')
    ax2.set_title('Executor Degradation & Adaptation (The Friction)')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Cross Entropy Loss')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_scratch_proof()