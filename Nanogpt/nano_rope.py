"""
=============================================================================
 nano_rope.py — RoFormer / Rotary Position Embedding (RoPE) vs learned
                positions, head to head on the same GPT
=============================================================================
 Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
 (Su et al., 2021, arXiv:2104.09864).

 THE IDEA IN ONE PARAGRAPH
 -------------------------
 GPT-2 ADDS a learned vector p_i to each token embedding — position is a
 thing the model must memorise, one slot per index, and position 65 is
 meaningless if you trained with 64. RoPE instead ROTATES each query and
 key by an angle proportional to its position:  q_m -> R(mθ)·q_m. Because
 rotations compose, the attention dot product q_m·k_n depends ONLY on the
 relative offset m−n (paper Eq. 16):
        (R_m W_q x_m)ᵀ (R_n W_k x_n) = xᵀ W_q R_{n−m} W_k x_n
 Position stops being a lookup table and becomes geometry. Consequences:
   * no positional parameters at all,
   * relative offsets are what attention sees (linguistically right),
   * any position index works — including ones never seen in training.
 This is why LLaMA, GPT-NeoX, Qwen, DeepSeek — essentially every modern
 open LLM — use RoPE.

 WHAT THIS SCRIPT DOES
 ---------------------
 1. Builds TWO GPTs identical except for how they encode position:
      A) learned absolute embeddings (your nanogpt_attention.py)
      B) RoPE applied to q,k inside every attention head
 2. Trains both on tiny-Shakespeare with the same budget; compares curves.
 3. THE PARTY TRICK — length extrapolation: both models train at context
    64, then we evaluate at 64/128/256. The learned-position model has no
    row 65 in its table (we show it fails by construction); RoPE just
    rotates further and keeps working (loss degrades gracefully).

 RUN IT
 ------
   python nano_rope.py            # ~20-30 min on CPU
   python nano_rope.py --quick    # ~4 min

 Writes runs/rope.json + runs/rope_extrapolation.png
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
# SECTION 0 — Configuration
# ---------------------------------------------------------------------------
# TRAIN context is deliberately short (64) so the extrapolation test at
# 128/256 means something. Both models share every other hyperparameter.
# ---------------------------------------------------------------------------
block_size = 64                               # training context length
EXTRA      = [64, 128, 256]                   # evaluation context lengths
batch_size = 32
n_embd     = 128 if not args.quick else 64
n_head     = 4
n_layer    = 4  if not args.quick else 2
dropout    = 0.1
max_iters  = 2500 if not args.quick else 300
eval_every = 250  if not args.quick else 100
eval_iters = 80   if not args.quick else 20
lr         = 3e-4

# ---------------------------------------------------------------------------
# SECTION 1 — Data (same tiny-Shakespeare pipeline as every nano script)
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
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split, T=block_size):
    """T is a parameter so the extrapolation probe can request longer rows."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - T, (batch_size,))
    x = torch.stack([d[i:i + T] for i in ix])
    y = torch.stack([d[i + 1:i + T + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# SECTION 2 — RoPE itself (paper Eq. 15 + the efficient Eq. 34)
# ---------------------------------------------------------------------------
# The rotation matrix R_m is block-diagonal: the head dimension is split
# into d/2 independent 2D planes, and plane i is rotated by angle m·θ_i,
#       θ_i = 10000^(−2i/d)                                   (paper §3.2.2)
# — a geometric ladder of frequencies exactly like the sinusoidal encoding,
# but MULTIPLIED into q,k instead of ADDED to the embedding.
# Never build the big matrix: Eq. 34 says rotating x is just
#       R_m x = x ⊙ cos(mθ) + rotate_half(x) ⊙ sin(mθ)
# where rotate_half pairs the dims: (x1,x2,x3,x4,…) -> (−x2,x1,−x4,x3,…).
# Cheap, exact, and differentiable.
# ---------------------------------------------------------------------------
def rope_cache(T, head_dim, device):
    """cos/sin tables of shape (T, head_dim) — computable for ANY T,
    which is precisely why RoPE extrapolates and lookup tables cannot."""
    theta = 10000.0 ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    m = torch.arange(T, device=device).float()
    ang = torch.outer(m, theta)                    # (T, head_dim/2)
    ang = torch.repeat_interleave(ang, 2, dim=-1)  # (T, head_dim) — θ1θ1θ2θ2…
    return ang.cos(), ang.sin()

def rotate_half(x):
    """(…, x1,x2,x3,x4,…) -> (…, −x2,x1,−x4,x3,…): the 90° partner
    that lets cos/sin products implement each 2×2 rotation block."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x, cos, sin):
    """x: (B, heads, T, head_dim) — rotate every position's q or k vector
    by its own angle. One fused multiply-add. (Paper Eq. 34.)"""
    T = x.shape[-2]
    return x * cos[:T] + rotate_half(x) * sin[:T]

# ---------------------------------------------------------------------------
# SECTION 3 — One attention block, two position modes
# ---------------------------------------------------------------------------
# mode="learned": position enters ONCE, added to the token embedding
#                 (the block itself is position-blind).
# mode="rope":    position enters INSIDE every head, rotating q and k just
#                 before the dot product. Values are NOT rotated — position
#                 should steer WHERE attention looks, not WHAT it copies.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.hs = n_embd // n_head
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ff = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q, k, v = (t.view(B, T, n_head, self.hs).transpose(1, 2) for t in (q, k, v))
        if self.mode == "rope":
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # <- RoPE
        att = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        return x + self.ff(self.ln2(x))

# ---------------------------------------------------------------------------
# SECTION 4 — The GPT, parameterised by position mode
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, mode):
        super().__init__()
        assert mode in ("learned", "rope")
        self.mode = mode
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        if mode == "learned":
            # a lookup table with exactly block_size rows — row 65 does
            # not exist, which is the whole extrapolation story
            self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(Block(mode) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = sin = None
        if self.mode == "learned":
            if T > block_size:
                raise ValueError(f"learned positions cannot exceed {block_size}")
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

# ---------------------------------------------------------------------------
# SECTION 5 — Train both models with the identical budget
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_loss(model, T=block_size):
    model.eval()
    ls = torch.zeros(eval_iters)
    for k in range(eval_iters):
        _, l = model(*get_batch("val", T))
        ls[k] = l.item()
    model.train()
    return ls.mean().item()

results = {}
for mode in ("learned", "rope"):
    torch.manual_seed(1337)                       # same init & data order
    model = GPT(mode).to(device)
    nparams = sum(p_.numel() for p_ in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = {"steps": [], "val": []}
    t0 = time.time()
    for it in range(max_iters + 1):
        if it % eval_every == 0:
            L = eval_loss(model)
            hist["steps"].append(it); hist["val"].append(L)
            print(f"[{mode:7s}] step {it:5d} | val {L:.4f} | {time.time()-t0:6.1f}s")
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    results[mode] = {"params": nparams, "history": hist,
                     "final_val": hist["val"][-1], "model": model}

# ---------------------------------------------------------------------------
# SECTION 6 — The extrapolation probe: evaluate BEYOND the training context
# ---------------------------------------------------------------------------
# Both models saw only 64-token windows in training. Now ask for 128 and
# 256. The learned model cannot even index positions ≥64 (we catch the
# error). RoPE simply computes cos/sin for the new positions — the
# rotation formula never heard of a maximum length.
# ---------------------------------------------------------------------------
print("\n================ LENGTH EXTRAPOLATION ================")
print(f"(both models trained at context {block_size})")
extrap = {}
for mode in ("learned", "rope"):
    extrap[mode] = {}
    for T in EXTRA:
        try:
            L = eval_loss(results[mode]["model"], T)
            extrap[mode][T] = L
            print(f"[{mode:7s}] context {T:4d} -> val loss {L:.4f}")
        except ValueError as e:
            extrap[mode][T] = None
            print(f"[{mode:7s}] context {T:4d} -> ✗ impossible ({e})")

# ---------------------------------------------------------------------------
# SECTION 7 — Plot + save
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, col in [("learned", "#EF5B7B"), ("rope", "#5BE3A6")]:
        h = results[mode]["history"]
        axes[0].plot(h["steps"], h["val"], color=col, lw=2, label=mode)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("val loss (nats/char)")
    axes[0].set_title("training: learned positions vs RoPE"); axes[0].legend()
    axes[0].grid(alpha=0.3)
    xs = [T for T in EXTRA]
    ro = [extrap["rope"][T] for T in EXTRA]
    axes[1].plot(xs, ro, "o-", color="#5BE3A6", lw=2, label="RoPE")
    le = [extrap["learned"][T] for T in EXTRA if extrap["learned"][T] is not None]
    axes[1].plot(xs[:len(le)], le, "o-", color="#EF5B7B", lw=2, label="learned")
    if len(le) < len(xs):
        axes[1].axvline(block_size, color="#EF5B7B", ls=":", lw=1.5)
        axes[1].text(block_size * 1.05, max(ro), "learned:\nimpossible past here",
                     color="#EF5B7B", fontsize=8)
    axes[1].set_xlabel("evaluation context length"); axes[1].set_ylabel("val loss")
    axes[1].set_title(f"extrapolation beyond training context ({block_size})")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "runs", "rope_extrapolation.png"), dpi=150)
    print("wrote runs/rope_extrapolation.png")
except Exception as e:
    print("plot skipped:", e)

json.dump({
    "name": "rope_vs_learned",
    "train_context": block_size,
    "learned": {"params": results["learned"]["params"],
                "final_val": results["learned"]["final_val"],
                "history": results["learned"]["history"],
                "extrapolation": extrap["learned"]},
    "rope":    {"params": results["rope"]["params"],
                "final_val": results["rope"]["final_val"],
                "history": results["rope"]["history"],
                "extrapolation": extrap["rope"]},
    "verdict_hint": "RoPE should match or beat learned positions at the "
                    "training length AND keep producing finite, gracefully "
                    "degrading loss at 2x-4x context, where learned "
                    "positions cannot run at all.",
}, open(os.path.join(HERE, "runs", "rope.json"), "w"), indent=2)
print("wrote runs/rope.json")
