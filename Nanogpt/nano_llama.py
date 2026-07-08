"""
=============================================================================
 nano_llama.py — the LLaMA recipe vs vanilla GPT-2, same budget, same data
=============================================================================
 Paper: "LLaMA: Open and Efficient Foundation Language Models"
 (Touvron et al., Meta AI 2023, arXiv:2302.13971).

 WHAT LLAMA ACTUALLY IS
 ----------------------
 Architecturally, LLaMA is a decoder-only Transformer with three
 borrowed upgrades over the GPT-2 design (paper §2.2):
   1. RMSNorm PRE-normalization  [from GPT-3's stability lessons]
   2. SwiGLU activation, width ⅔·4d  [from PaLM]
   3. Rotary embeddings (RoPE)  [from GPTNeo / Su et al. 2021]
 Its real contribution is a TRAINING PHILOSOPHY: Chinchilla says a 10B
 model is compute-OPTIMAL near 200B tokens, but optimal-for-training is
 not optimal-for-USING. LLaMA trains "small" models far PAST their
 Chinchilla point (7B on 1000B tokens = 143 tokens/param, 7x Chinchilla's
 20) because a model is trained once and inferenced forever: LLaMA-13B
 beat GPT-3 175B while being 13x cheaper to run.

 WHAT THIS SCRIPT DOES
 ---------------------
 1. Builds TWO models with matched parameter budgets:
      A) vanilla nanoGPT   (LayerNorm + ReLU-MLP + learned positions)
      B) nano-LLaMA        (RMSNorm + SwiGLU(⅔·4d) + RoPE, no biases)
 2. Trains both with the same step budget on tiny-Shakespeare, with
    LLaMA's optimizer recipe (AdamW β2=0.95, cosine schedule, warmup,
    grad clipping) applied to BOTH so only the architecture differs.
 3. Compares the loss curves and generates a sample from each.

 WHAT TO EXPECT AT THIS SCALE
 ----------------------------
 The three upgrades are worth a few percent of loss at 100M+ parameters;
 at nano scale expect nano differences — LLaMA-arch usually edges ahead,
 but the honest claim is "matches or slightly beats the GPT-2 recipe with
 FEWER positional parameters and better long-context behaviour" (see
 nano_rope.py for that half of the story).

 RUN IT
 ------
   python nano_llama.py            # ~25-35 min on CPU
   python nano_llama.py --quick    # ~4 min

 Writes runs/llama.json + runs/llama_vs_gpt.png
=============================================================================
"""

import argparse, json, math, os, sys, time, urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

if hasattr(sys.stdout, "reconfigure"):        # Windows cp1252 console fix
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
args = p.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# SECTION 0 — Configuration (LLaMA's optimizer recipe, miniaturised)
# ---------------------------------------------------------------------------
block_size = 128 if not args.quick else 64
batch_size = 32
n_embd     = 128 if not args.quick else 64
n_head     = 4
n_layer    = 4  if not args.quick else 2
dropout    = 0.1
max_iters  = 3000 if not args.quick else 300
eval_every = 250  if not args.quick else 100
eval_iters = 80   if not args.quick else 20
# LLaMA §2.3: AdamW β=(0.9, 0.95), cosine to 10% of peak, weight decay 0.1,
# gradient clipping 1.0, warmup steps.
lr, lr_final_frac = 3e-4, 0.10
betas, weight_decay, grad_clip = (0.9, 0.95), 0.1, 1.0
warmup = 100 if not args.quick else 20

# ---------------------------------------------------------------------------
# SECTION 1 — Data
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "input.txt")
if not os.path.exists(DATA):
    print("downloading tiny-shakespeare …")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt", DATA)
text = open(DATA, encoding="utf-8").read()
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
decode = lambda t: "".join(itos[i] for i in t)
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# SECTION 2 — Upgrade #1: RMSNorm (Zhang & Sennrich 2019; LLaMA §2.2)
# ---------------------------------------------------------------------------
#   RMSNorm(x) = x / RMS(x) · g,     RMS(x) = sqrt(mean(x²) + ε)
# LayerNorm subtracts the mean AND divides by the std, with a bias and a
# gain. RMSNorm drops the mean-centering and the bias: re-SCALING is what
# actually stabilises training; re-CENTERING is dead weight. ~10-40% cheaper
# and — applied PRE-sublayer — the key to stable very deep stacks.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.g

# ---------------------------------------------------------------------------
# SECTION 3 — Upgrade #2: SwiGLU feed-forward (Shazeer 2020; LLaMA §2.2)
# ---------------------------------------------------------------------------
#   FFN_SwiGLU(x) = W₂ · [ SiLU(W₁x) ⊙ (W₃x) ],   SiLU(z) = z·σ(z)
# A GATED MLP: one branch computes content, the other computes a gate that
# multiplies it — the network learns per-feature "how much of this should
# pass". Three matrices instead of two, so LLaMA shrinks the hidden width
# from 4d to ⅔·4d ≈ 2.67d to keep the parameter count identical.
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = int(2 * 4 * dim / 3)                 # ⅔ · 4d, LLaMA's choice
        self.w1 = nn.Linear(dim, hidden, bias=False)  # content branch
        self.w3 = nn.Linear(dim, hidden, bias=False)  # gate branch
        self.w2 = nn.Linear(hidden, dim, bias=False)  # back down
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ---------------------------------------------------------------------------
# SECTION 4 — Upgrade #3: RoPE (Su et al. 2021; full story in nano_rope.py)
# ---------------------------------------------------------------------------
def rope_cache(T, head_dim, device):
    theta = 10000.0 ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    ang = torch.outer(torch.arange(T, device=device).float(), theta)
    ang = torch.repeat_interleave(ang, 2, dim=-1)
    return ang.cos(), ang.sin()

def rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x, cos, sin):
    T = x.shape[-2]
    return x * cos[:T] + rotate_half(x) * sin[:T]

# ---------------------------------------------------------------------------
# SECTION 5 — Two blocks: GPT-2 recipe vs LLaMA recipe
# ---------------------------------------------------------------------------
# Same attention core; the differences are exactly the three upgrades:
#             GPT-2 block                LLaMA block
#   norm      LayerNorm (pre)            RMSNorm (pre)
#   FFN       Linear-ReLU-Linear (4d)    SwiGLU (⅔·4d, gated)
#   position  learned table (in model)   RoPE (inside attention)
#   biases    yes                        none anywhere
# ---------------------------------------------------------------------------
class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.hs = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ff = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)

    def attn(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, n_head, self.hs).transpose(1, 2) for t in (q, k, v))
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        return (F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)

    def forward(self, x, cos=None, sin=None):
        x = x + self.drop(self.proj(self.attn(self.ln1(x), cos, sin)))
        return x + self.ff(self.ln2(x))

class LlamaBlock(GPTBlock):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)   # no biases
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.ff = SwiGLU(n_embd)                               # gated FFN
        self.ln1, self.ln2 = RMSNorm(n_embd), RMSNorm(n_embd)  # RMS pre-norm

# ---------------------------------------------------------------------------
# SECTION 6 — The two models
# ---------------------------------------------------------------------------
class LM(nn.Module):
    def __init__(self, arch):
        super().__init__()
        assert arch in ("gpt2", "llama")
        self.arch = arch
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        if arch == "gpt2":
            self.pos_emb = nn.Embedding(block_size, n_embd)
            self.blocks = nn.ModuleList(GPTBlock() for _ in range(n_layer))
            self.ln_f = nn.LayerNorm(n_embd)
        else:
            self.blocks = nn.ModuleList(LlamaBlock() for _ in range(n_layer))
            self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=(arch == "gpt2"))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = sin = None
        if self.arch == "gpt2":
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        else:
            cos, sin = rope_cache(T, n_embd // n_head, idx.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# ---------------------------------------------------------------------------
# SECTION 7 — Train both with LLaMA's optimizer recipe
# ---------------------------------------------------------------------------
def cosine_lr(it):
    """LLaMA §2.3: linear warmup, then cosine decay to 10% of peak."""
    if it < warmup:
        return lr * it / warmup
    t = (it - warmup) / max(1, max_iters - warmup)
    return lr * (lr_final_frac + (1 - lr_final_frac) * 0.5 * (1 + math.cos(math.pi * t)))

@torch.no_grad()
def eval_loss(model):
    model.eval()
    ls = torch.zeros(eval_iters)
    for k in range(eval_iters):
        _, l = model(*get_batch("val"))
        ls[k] = l.item()
    model.train()
    return ls.mean().item()

results = {}
for arch in ("gpt2", "llama"):
    torch.manual_seed(1337)
    model = LM(arch).to(device)
    nparams = sum(p_.numel() for p_ in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas,
                            weight_decay=weight_decay)
    hist = {"steps": [], "val": []}
    t0 = time.time()
    for it in range(max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(it)
        if it % eval_every == 0:
            L = eval_loss(model)
            hist["steps"].append(it); hist["val"].append(L)
            print(f"[{arch:5s}] step {it:5d} | val {L:.4f} | lr {cosine_lr(it):.2e} "
                  f"| {time.time()-t0:6.1f}s")
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # LLaMA clip 1.0
        opt.step()
    results[arch] = {"params": nparams, "history": hist,
                     "final_val": hist["val"][-1], "model": model}
    print(f"[{arch:5s}] params={nparams:,}  final val={hist['val'][-1]:.4f}")

# ---------------------------------------------------------------------------
# SECTION 8 — Samples, plot, save
# ---------------------------------------------------------------------------
print("\n----- 200-char samples -----")
for arch in ("gpt2", "llama"):
    m = results[arch]["model"]; m.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(f"[{arch}] {decode(m.generate(ctx, 200)[0].tolist())!r}\n")

os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arch, col in [("gpt2", "#EF5B7B"), ("llama", "#5BE3A6")]:
        h = results[arch]["history"]
        ax.plot(h["steps"], h["val"], color=col, lw=2,
                label=f"{arch} ({results[arch]['params']:,} params)")
    ax.set_xlabel("step"); ax.set_ylabel("val loss (nats/char)")
    ax.set_title("GPT-2 recipe vs LLaMA recipe — same data, steps, optimizer")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "runs", "llama_vs_gpt.png"), dpi=150)
    print("wrote runs/llama_vs_gpt.png")
except Exception as e:
    print("plot skipped:", e)

json.dump({
    "name": "llama_vs_gpt2",
    "gpt2":  {"params": results["gpt2"]["params"],
              "final_val": results["gpt2"]["final_val"],
              "history": results["gpt2"]["history"]},
    "llama": {"params": results["llama"]["params"],
              "final_val": results["llama"]["final_val"],
              "history": results["llama"]["history"]},
    "recipe": {"norm": "RMSNorm pre-norm", "ffn": "SwiGLU 2/3·4d",
               "position": "RoPE", "biases": "none",
               "optimizer": "AdamW b2=0.95, cosine->10%, clip 1.0"},
    "verdict_hint": "Expect llama <= gpt2 in val loss at equal params/steps; "
                    "the recipe's big wins (stability at depth, long-context "
                    "RoPE) grow with scale.",
}, open(os.path.join(HERE, "runs", "llama.json"), "w"), indent=2)
print("wrote runs/llama.json")
