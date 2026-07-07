"""
=============================================================================
 nanogpt_attention.py — a GPT built around ONE idea: causal self-attention
=============================================================================
 Paper: "Attention Is All You Need" (Vaswani et al., 2017), decoder-only
 variant as popularised by GPT-2 and Karpathy's nanoGPT.

 WHAT THIS SCRIPT IS
 -------------------
 The complete, runnable training script distilled from your DIY_GPT_Dev.ipynb:
 a character-level GPT trained on tiny-Shakespeare whose ONLY mixing mechanism
 across time is self-attention. Every section is labelled and explained.

 RUN IT
 ------
   python nanogpt_attention.py            # full run (~15-25 min on CPU)
   python nanogpt_attention.py --quick    # smoke test (~2 min on CPU)

 It writes:
   runs/gpt_attention.json   — loss curves (for compare.py)
   runs/gpt_attention.pt     — model weights
   and prints a 500-character Shakespeare sample at the end.
=============================================================================
"""

import argparse, json, math, os, time, urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# SECTION 0 — Configuration
# ---------------------------------------------------------------------------
# One dataclass-like namespace holds every knob. The sizes below are chosen
# so the model trains to recognisable Shakespeare on a CPU. n_embd/n_head/
# n_layer are the three numbers that set the parameter count:
#     N ≈ 12 · n_layer · n_embd²      (the same formula the scaling-laws
#                                      paper uses — see nano_scaling_laws.py)
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true", help="tiny smoke-test run")
args = p.parse_args()

torch.manual_seed(1337)                      # reproducibility
device     = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 128 if not args.quick else 64   # context length (tokens the model can see)
batch_size = 32  if not args.quick else 16   # independent sequences per step
n_embd     = 128 if not args.quick else 64   # width of the residual stream
n_head     = 4                               # attention heads (n_embd/n_head each)
n_layer    = 4  if not args.quick else 2     # stacked Transformer blocks
dropout    = 0.1
max_iters  = 3000 if not args.quick else 300
eval_every = 250  if not args.quick else 100
eval_iters = 100  if not args.quick else 20
lr         = 3e-4                            # AdamW learning rate

# ---------------------------------------------------------------------------
# SECTION 1 — Data: tiny-Shakespeare, character-level
# ---------------------------------------------------------------------------
# We tokenize at the CHARACTER level: the vocabulary is just the ~65 distinct
# characters in the file. This keeps the embedding table tiny so virtually all
# parameters live in the Transformer itself — which is what we want to study.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "input.txt")
if not os.path.exists(DATA):
    print("downloading tiny-shakespeare …")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt", DATA)
text = open(DATA, encoding="utf-8").read()

chars      = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}    # char -> int
itos = {i: c for i, c in enumerate(chars)}    # int  -> char
encode = lambda s: [stoi[c] for c in s]
decode = lambda t: "".join(itos[i] for i in t)

data  = torch.tensor(encode(text), dtype=torch.long)
n     = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]     # 90/10 split

def get_batch(split):
    """Sample a random (x, y) batch. y is x shifted one character left:
    at every position the model's job is to predict the NEXT character."""
    d  = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i + block_size]     for i in ix])
    y  = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# SECTION 2 — THE self-attention head (the heart of the file)
# ---------------------------------------------------------------------------
# One head does exactly this, for every position t in the sequence:
#   1. q_t = x_t W_Q        "what am I looking for?"
#   2. k_s = x_s W_K        "what do I contain?"      (for every position s)
#   3. score(t,s) = q_t·k_s / sqrt(d)   affinity between t and every s
#   4. CAUSAL MASK: positions s > t are set to -inf — a character may not
#      look at the future. This single line is what makes the model
#      autoregressive (and is exactly the line BERT deletes — see nanobert.py)
#   5. softmax over s -> attention weights (a probability distribution)
#   6. out_t = Σ_s weights(t,s) · v_s   a weighted average of value vectors
# The sqrt(d) scaling keeps the softmax from saturating at init: without it,
# dot products grow with dimension and the weights collapse to one-hot.
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 'tril' is the causal mask: 1s on/below the diagonal, 0s above.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                   # (B,T,hs)
        q = self.query(x)                                 # (B,T,hs)
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B,T,T) scores
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal!
        att = F.softmax(att, dim=-1)                      # rows sum to 1
        att = self.dropout(att)
        v = self.value(x)                                 # (B,T,hs)
        return att @ v                                    # (B,T,hs)

# ---------------------------------------------------------------------------
# SECTION 3 — Multi-head attention: several heads in parallel
# ---------------------------------------------------------------------------
# Each head gets n_embd/n_head dimensions and learns a DIFFERENT notion of
# relevance (one head may track "previous vowel", another "opening quote").
# Their outputs are concatenated back to n_embd and mixed by a projection.
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(Head(head_size) for _ in range(n_head))
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,n_embd)
        return self.dropout(self.proj(out))

# ---------------------------------------------------------------------------
# SECTION 4 — Feed-forward: per-position computation
# ---------------------------------------------------------------------------
# Attention MOVES information between positions; the MLP THINKS about it.
# Applied to every position independently, with the classic 4x expansion.
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

# ---------------------------------------------------------------------------
# SECTION 5 — The Transformer block: communicate, then compute
# ---------------------------------------------------------------------------
# Two residual ("x + …") connections make deep stacks trainable: gradients
# always have a clean path back through the additions. LayerNorm is applied
# BEFORE each sub-layer (pre-norm, the modern GPT-2 arrangement).
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa  = MultiHeadAttention()
        self.ff  = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # communicate across positions
        x = x + self.ff(self.ln2(x))   # compute at each position
        return x

# ---------------------------------------------------------------------------
# SECTION 6 — The GPT: embeddings -> blocks -> language-model head
# ---------------------------------------------------------------------------
# token embedding: WHAT each character is.
# position embedding: WHERE it sits (attention alone is permutation-blind —
#   without positions, "dog bites man" = "man bites dog").
# lm_head: projects the final n_embd vector to vocab_size logits = a score
#   for every possible next character.
# Loss: cross-entropy between logits and the actual next character — the
#   same L (in nats/token) that the scaling-laws papers measure.
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)                                   # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))    # (T,C)
        x = self.blocks(tok + pos)
        logits = self.lm_head(self.ln_f(x))                       # (B,T,vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressive sampling: feed the context, take the logits of the
        LAST position, sample one character, append, repeat. This loop is
        possible only because of the causal mask — the model was trained
        never to peek ahead, so it knows how to run one step at a time."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# ---------------------------------------------------------------------------
# SECTION 7 — Training loop with periodic evaluation
# ---------------------------------------------------------------------------
model = GPT().to(device)
n_params = sum(p.numel() for p in model.parameters())
n_nonemb = n_params - model.tok_emb.weight.numel() - model.pos_emb.weight.numel()
print(f"device={device}  params={n_params:,}  non-embedding={n_nonemb:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
    """Average the loss over many random batches — a single batch is far too
    noisy to compare models with."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

history = {"steps": [], "train": [], "val": []}
t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_every == 0:
        L = estimate_loss()
        history["steps"].append(it)
        history["train"].append(L["train"])
        history["val"].append(L["val"])
        print(f"step {it:5d} | train {L['train']:.4f} | val {L['val']:.4f} "
              f"| {time.time()-t0:6.1f}s")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------------------------------------------------------------------
# SECTION 8 — Save results (for compare.py) and generate a sample
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
result = {
    "name": "gpt_attention", "objective": "next-token (causal LM)",
    "params": n_params, "non_embedding_params": n_nonemb,
    "block_size": block_size, "n_layer": n_layer, "n_embd": n_embd,
    "final_val_loss_nats": history["val"][-1], "history": history,
    # C = 6·N·B·S — total training FLOPs, the currency of both scaling papers
    "train_flops_6NBS": 6 * n_nonemb * batch_size * block_size * max_iters,
}
json.dump(result, open(os.path.join(HERE, "runs", "gpt_attention.json"), "w"), indent=2)
torch.save(model.state_dict(), os.path.join(HERE, "runs", "gpt_attention.pt"))

print("\n----- sample (500 chars) -----")
ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(ctx, 500)[0].tolist()))
print("\nwrote runs/gpt_attention.json + .pt")
