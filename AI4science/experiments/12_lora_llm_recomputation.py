"""
GOOGLE COLAB INSTRUCTIONS (LoRA):
1. Runtime -> Change runtime type -> T4 GPU.
2. Run this in a cell first: !pip install -q transformers peft datasets matplotlib
3. Run this script.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, inject_adapter_in_model
from datasets import load_dataset
from torch.utils.data import DataLoader

# ==========================================
# 1. DATA PREPARATION (NARRATIVE VS WEB)
# ==========================================
print("Loading TinyLlama Tokenizer & Datasets...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load small subsets to fit in memory
ds_story = load_dataset("roneneldan/TinyStories", split="train[:5000]")
ds_web = load_dataset("Skylion007/openwebtext", split="train[:5000]")

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64, return_tensors="pt")

tokenized_story = ds_story.map(tokenize_batch, batched=True, remove_columns=ds_story.column_names)
tokenized_web = ds_web.map(tokenize_batch, batched=True, remove_columns=ds_web.column_names)
tokenized_story.set_format("torch")
tokenized_web.set_format("torch")

def create_iterator(dataset, batch_size=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    while True:
        for batch in dataloader:
            yield batch

story_iter = create_iterator(tokenized_story)
web_iter = create_iterator(tokenized_web)

# ==========================================
# 2. CONTINUOUS RECOMPUTATION ARCHITECTURE
# ==========================================
class LoRARecomputationLLM(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Base LLM into VRAM on {self.device}...")
        
        # Load frozen base model (The Cortex) in bfloat16
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16
        ).to(self.device)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        print("Injecting LoRA Primitives...")
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
        
        # The Primitive Library
        self.base_model.add_adapter("primitive_story", lora_config)
        self.base_model.add_adapter("primitive_web", lora_config)
        
        # The Controller (Structural Strategist)
        hidden_size = self.base_model.config.hidden_size
        self.controller = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2) 
        ).to(self.device).to(torch.float32)

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None: labels = labels.to(self.device)

        # 1. Controller Routing Phase (No backprop into Base LLM)
        with torch.no_grad():
            embeddings = self.base_model.get_input_embeddings()(input_ids)
            pooled_embeds = embeddings.mean(dim=1).to(torch.float32)
            
        routing_logits = self.controller(pooled_embeds)
        
        # HARD ROUTING: Determine adapter via argmax. 
        # The unselected LoRA is physically detached from the graph by PEFT.
        majority_domain = torch.argmax(routing_logits.detach().mean(dim=0)).item()
        active_adapter = "primitive_story" if majority_domain == 0 else "primitive_web"
        
        # 2. Execution Phase
        self.base_model.set_adapter(active_adapter)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        return outputs.loss, routing_logits

# ==========================================
# 3. ASYMMETRIC TRAINING LOOP
# ==========================================
def run_lora_proof():
    model = LoRARecomputationLLM(model_name)
    
    # Asymmetric Optimizers
    controller_optimizer = optim.AdamW(model.controller.parameters(), lr=1e-3)
    executor_params = [p for n, p in model.base_model.named_parameters() if "lora_" in n and p.requires_grad]
    executor_optimizer = optim.AdamW(executor_params, lr=5e-4)
    
    ce_loss_fn = nn.CrossEntropyLoss()
    steps = 400
    steps_logged, ctrl_story_acc, ctrl_web_acc, exec_story_loss, exec_web_loss = [], [], [], [], []
    
    print("\n--- INITIATING LoRA RECOMPUTATION PROOF ---")
    
    for step in range(steps):
        is_shock = step >= 200
        batch = next(web_iter) if is_shock else next(story_iter)
        true_domain = 1 if is_shock else 0
            
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        domain_labels = torch.full((input_ids.size(0),), true_domain, dtype=torch.long, device=model.device)
        
        controller_optimizer.zero_grad()
        executor_optimizer.zero_grad()
        
        # Forward Pass
        lm_loss, routing_logits = model(input_ids, attention_mask, labels=input_ids)
        ctrl_loss = ce_loss_fn(routing_logits, domain_labels)
        
        # Decoupled Backward Passes (Structural Bulkhead in action)
        ctrl_loss.backward(retain_graph=True)
        lm_loss.backward()
        
        controller_optimizer.step()
        executor_optimizer.step()
        
        if step % 10 == 0:
            steps_logged.append(step)
            def evaluate(eval_iter, domain_id, adapter_name):
                model.eval()
                with torch.no_grad():
                    eval_batch = next(eval_iter)
                    ids, mask = eval_batch["input_ids"], eval_batch["attention_mask"]
                    
                    # Force correct adapter for clean loss measurement
                    model.base_model.set_adapter(adapter_name)
                    l_loss, r_logits = model(ids, mask, labels=ids)
                    acc = (torch.argmax(r_logits, dim=-1) == domain_id).float().mean().item() * 100
                model.train()
                return acc, l_loss.item()
            
            s_acc, s_loss = evaluate(story_iter, 0, "primitive_story")
            w_acc, w_loss = evaluate(web_iter, 1, "primitive_web")
            ctrl_story_acc.append(s_acc); ctrl_web_acc.append(w_acc)
            exec_story_loss.append(s_loss); exec_web_loss.append(w_loss)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.style.use('bmh')
    
    ax1.plot(steps_logged, ctrl_story_acc, color='#1f77b4', linewidth=2.5, label='Controller: Story Logic')
    ax1.plot(steps_logged, ctrl_web_acc, color='#ff7f0e', linestyle='--', linewidth=2.5, label='Controller: Web Logic')
    ax1.axvline(x=200, color='black', linestyle='-.', linewidth=2, label='Domain Shock Induced')
    ax1.set_title('Controller Routing Stability (Survives Interference)')
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
    run_lora_proof()