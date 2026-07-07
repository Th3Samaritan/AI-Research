"""
=============================================================================
 nano_scaling_laws.py — reproduce Kaplan et al. (2020) at kitchen scale
=============================================================================
 Paper: "Scaling Laws for Neural Language Models" (arXiv:2001.08361).

 THE CLAIM WE ARE TESTING
 ------------------------
 The paper says: when data is plentiful, the converged test loss of a
 Transformer depends on its NON-EMBEDDING parameter count N as a power law

        L(N) = (Nc / N)^alpha_N          (their Eq. 1.1)

 with alpha_N ≈ 0.076 on WebText2. A power law means: every doubling of N
 buys the SAME multiplicative reduction in loss — a straight line on a
 log-log plot.

 WHAT THIS SCRIPT DOES
 ---------------------
 1. Trains a LADDER of GPTs (same architecture as nanogpt_attention.py,
    increasing width) on tiny-Shakespeare, each to near-convergence.
 2. Records the best validation loss for each size.
 3. Fits BOTH functional forms used by the two papers:
      pure power law   L = (Nc/N)^a          (Kaplan)
      saturating form  L = E + A/N^a         (Chinchilla Eq. 2, D-term absent)
    The second fits small-scale data better because char-level Shakespeare
    has a high irreducible entropy E that tiny models already approach.
 4. Plots the ladder on log-log axes and saves everything for compare.py.

 WHAT TO EXPECT AT THIS SCALE (honesty section)
 ----------------------------------------------
 * You WILL see the qualitative claim: loss falls smoothly and predictably
   with N, no cliffs, no luck — the central message of the paper.
 * Your fitted exponent will NOT be 0.076. Exponents are dataset- and
   tokenizer-dependent (char-level ≠ BPE WebText2), and our largest model
   may start to be data-bound on ~1M characters (that bending toward a
   floor is exactly the L(N,D) story — and the Chinchilla script's topic).

 RUN IT
 ------
   python nano_scaling_laws.py            # ~30-45 min on CPU (5 models)
   python nano_scaling_laws.py --quick    # ~5 min, 4 tiny models

 Writes runs/scaling_laws.json + runs/scaling_laws.png
=============================================================================
"""

import argparse, json, math, os, sys, time, urllib.request
import torch

if hasattr(sys.stdout, "reconfigure"):        # Windows cp1252 console fix
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch.nn as nn
from torch.nn import functional as F

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
args = p.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# SECTION 0 — The experiment grid
# ---------------------------------------------------------------------------
# The LADDER: widths chosen so non-embedding params span ~2 orders of
# magnitude. Depth is held at 2 and only width grows — the paper's Figure 5
# shows shape barely matters at fixed N ("scale beats shape"), which lets us
# get away with varying one knob.
# Every model gets the SAME data, batch size, and enough steps to be near
# convergence (more steps for bigger models would only sharpen the trend).
# ---------------------------------------------------------------------------
if args.quick:
    WIDTHS   = [16, 32, 64, 96]
    MAX_ITERS = 600
else:
    WIDTHS   = [16, 32, 64, 128, 192]
    MAX_ITERS = 2500

n_layer, n_head_max = 2, 4
block_size, batch_size = 64, 32
lr, eval_iters = 1e-3, 60

# ---------------------------------------------------------------------------
# SECTION 1 — Data (identical to nanogpt_attention.py)
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

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# SECTION 2 — A width-parameterised GPT
# ---------------------------------------------------------------------------
# Same model as nanogpt_attention.py, but n_embd is a constructor argument
# so the ladder can instantiate any size. (Fused-QKV formulation — computes
# the identical math to the per-Head version, just in fewer tensors.)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head, self.hs = n_head, n_embd // n_head
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ff = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                nn.Linear(4 * n_embd, n_embd))
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # reshape to (B, heads, T, head_size)
        q, k, v = (t.view(B, T, self.n_head, self.hs).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))   # causal
        y = F.softmax(att, dim=-1) @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

def non_embedding_params(m):
    """N as the paper defines it: EXCLUDING token & positional embeddings.
    The paper shows trends only come out clean with this definition
    (their Figure 6) — with embeddings included, small models look
    artificially big."""
    total = sum(p.numel() for p in m.parameters())
    return total - m.tok_emb.weight.numel() - m.pos_emb.weight.numel()

# ---------------------------------------------------------------------------
# SECTION 3 — Train the ladder
# ---------------------------------------------------------------------------
@torch.no_grad()
def val_loss(model):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        _, l = model(*get_batch("val"))
        losses[k] = l.item()
    model.train()
    return losses.mean().item()

ladder = []
for width in WIDTHS:
    n_head = min(n_head_max, max(1, width // 16))
    model = GPT(width, n_head).to(device)
    N = non_embedding_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float("inf")
    t0 = time.time()
    for it in range(MAX_ITERS):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (it + 1) % (MAX_ITERS // 5) == 0:
            best = min(best, val_loss(model))   # track best = "early stopping"
    ladder.append({"n_embd": width, "N_non_embedding": N, "best_val_loss": best})
    print(f"width {width:4d} | N={N:9,} | best val loss {best:.4f} "
          f"| {time.time()-t0:6.1f}s")

# ---------------------------------------------------------------------------
# SECTION 4 — Fit the two competing functional forms
# ---------------------------------------------------------------------------
# (a) Kaplan pure power law:  L = (Nc/N)^a
#     Taking logs: log L = a·log Nc − a·log N — LINEAR in log N, so it's a
#     one-line least-squares fit.
# (b) Chinchilla-style saturating law:  L = E + A/N^a
#     Has an IRREDUCIBLE floor E (the entropy of the text). We fit by a
#     small grid/least-squares over E: for each candidate E, (log(L−E) vs
#     log N) is again linear; pick the E with the best residual.
# ---------------------------------------------------------------------------
import numpy as np
Ns = np.array([r["N_non_embedding"] for r in ladder], dtype=float)
Ls = np.array([r["best_val_loss"]   for r in ladder], dtype=float)

# (a) pure power law
A_ = np.vstack([np.log(Ns), np.ones_like(Ns)]).T
(slope, intercept), *_ = np.linalg.lstsq(A_, np.log(Ls), rcond=None)
alpha_pure = -slope
Nc_pure = math.exp(intercept / alpha_pure) if alpha_pure != 0 else float("nan")
pred_pure = np.exp(A_ @ np.array([slope, intercept]))
r2_pure = 1 - np.sum((Ls - pred_pure) ** 2) / np.sum((Ls - Ls.mean()) ** 2)

# (b) saturating law — grid over the floor E
best_fit = None
for E in np.linspace(0.0, Ls.min() * 0.98, 200):
    y = np.log(Ls - E)
    (s, b), *_ = np.linalg.lstsq(A_, y, rcond=None)
    resid = np.sum((y - (A_ @ np.array([s, b]))) ** 2)
    if best_fit is None or resid < best_fit[0]:
        best_fit = (resid, E, -s, math.exp(b))
_, E_fit, alpha_sat, Afit = best_fit
pred_sat = E_fit + Afit / Ns ** alpha_sat
r2_sat = 1 - np.sum((Ls - pred_sat) ** 2) / np.sum((Ls - Ls.mean()) ** 2)

print("\n================ FITS ================")
print(f"Kaplan pure power law : L=(Nc/N)^a      a={alpha_pure:.3f}  R²={r2_pure:.4f}")
print(f"  (paper's WebText2 value: a=0.076 — different data => different a)")
print(f"Saturating law        : L=E+A/N^a       E={E_fit:.3f}  a={alpha_sat:.3f}  R²={r2_sat:.4f}")
print(f"  E is the fitted 'entropy of Shakespeare-chars' floor.")
print("The KEY qualitative check: is loss vs N a smooth, monotone power-ish")
print("curve with no jumps? That is the paper's core claim at any scale.")

# ---------------------------------------------------------------------------
# SECTION 5 — Plot + save
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.logspace(np.log10(Ns.min() * 0.8), np.log10(Ns.max() * 1.3), 100)
    ax.loglog(Ns, Ls, "o", ms=8, color="#F5A524", label="trained models")
    ax.loglog(xs, np.exp(slope * np.log(xs) + intercept), "--", color="#5B8DEF",
              label=f"pure power law  a={alpha_pure:.3f}")
    ax.loglog(xs, E_fit + Afit / xs ** alpha_sat, "-", color="#5BE3A6",
              label=f"L=E+A/N^a  E={E_fit:.2f}, a={alpha_sat:.2f}")
    ax.set_xlabel("N — non-embedding parameters")
    ax.set_ylabel("best val loss (nats/char)")
    ax.set_title("nano scaling law: loss vs model size (tiny-Shakespeare)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "runs", "scaling_laws.png"), dpi=150)
    print("wrote runs/scaling_laws.png")
except Exception as e:
    print("plot skipped:", e)

json.dump({
    "name": "scaling_laws_ladder",
    "ladder": ladder,
    "fit_pure_power": {"alpha": alpha_pure, "Nc": Nc_pure, "r2": r2_pure,
                       "paper_alpha_webtext2": 0.076},
    "fit_saturating": {"E": E_fit, "A": Afit, "alpha": alpha_sat, "r2": r2_sat},
    "verdict_hint": "Smooth monotone power-law-ish decline of L with N "
                    "reproduces Kaplan's qualitative claim; exact exponents "
                    "are dataset/tokenizer-specific.",
}, open(os.path.join(HERE, "runs", "scaling_laws.json"), "w"), indent=2)
print("wrote runs/scaling_laws.json")
