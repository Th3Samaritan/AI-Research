"""
=============================================================================
 nano_chinchilla.py — reproduce Hoffmann et al. (2022) "Chinchilla" at
                      kitchen scale: the IsoFLOP experiment
=============================================================================
 Paper: "Training Compute-Optimal Large Language Models" (arXiv:2203.15556).

 THE ARGUMENT WITH KAPLAN (why this paper exists)
 ------------------------------------------------
 Kaplan et al. (2020) said: given more compute, spend nearly all of it on a
 BIGGER MODEL —  N_opt ∝ C^0.73, data only D ∝ C^0.27.
 Chinchilla re-ran the experiment with a crucial fix (tune the learning-rate
 schedule to the training length, don't reuse one schedule for all runs) and
 found instead:
        N_opt ∝ C^0.50      D_opt ∝ C^0.50        (Table 2)
 i.e. model size and data should grow in EQUAL proportion — roughly
 "20 tokens per parameter". This is why Chinchilla (70B, 1.4T tokens)
 beat Gopher (280B, 300B tokens) using the SAME compute.

 WHAT THIS SCRIPT DOES — the paper's Approach 2 (IsoFLOP profiles) — plus
 its Approach 3 (parametric fit), both miniaturised:
 ------------------------------------------------------------------------
 1. Pick a handful of FIXED compute budgets C (measured, as both papers do,
    with the approximation  C = 6·N·D  FLOPs, D = tokens processed).
 2. For each budget, train SEVERAL model sizes N, giving each one exactly
    D = C/(6N) tokens: big model = few tokens, small model = many tokens.
 3. Each IsoFLOP curve (loss vs N at fixed C) shows a VALLEY: too small a
    model can't learn enough, too big a model is undertrained. Fit a
    parabola in log N to locate the valley bottom N_opt(C).      [Approach 2]
 4. Fit N_opt ∝ C^a across budgets — Chinchilla says a ≈ 0.5.
 5. Fit the parametric loss  L(N,D) = E + A/N^α + B/D^β  to ALL runs with a
    Huber loss (the paper's Eq. 2/10; they got E=1.69, A=406.4, B=410.7,
    α=0.34, β=0.28 on MassiveText), and derive the closed-form frontier
        a = β/(α+β),   b = α/(α+β).                              [Approach 3]

 WHAT TO EXPECT AT THIS SCALE (honesty section)
 ----------------------------------------------
 * The IsoFLOP VALLEYS are robust and you should see them clearly — the
   qualitative heart of the paper survives miniaturisation.
 * The fitted exponent a will be noisy with only 3-4 budgets of this size —
   treat "a is much closer to 0.5 than to 0.73" as the pass/fail question.
 * Tokens-per-parameter at the valley will NOT be 20 — that constant is
   specific to their data/tokenizer; the SHAPE of the answer is the point.

 RUN IT
 ------
   python nano_chinchilla.py            # ~40-60 min on CPU
   python nano_chinchilla.py --quick    # ~8 min, smaller grid

 Writes runs/chinchilla.json + runs/chinchilla_isoflop.png
=============================================================================
"""

import argparse, json, math, os, sys, time, urllib.request
import numpy as np

if hasattr(sys.stdout, "reconfigure"):        # Windows cp1252 console fix
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn
from torch.nn import functional as F

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
args = p.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# SECTION 0 — The IsoFLOP grid
# ---------------------------------------------------------------------------
# Budgets C in FLOPs (via C = 6·N·D). Consecutive budgets are ~4x apart so
# the fitted exponent has some lever arm. Model widths give N spanning ~30x.
# For each (C, width) pair the token allowance D — and hence the number of
# optimizer steps D/(batch·block) — is DERIVED, not chosen. That inversion
# ("compute is the budget, steps are the consequence") IS the experiment.
# ---------------------------------------------------------------------------
if args.quick:
    BUDGETS = [3e11, 1.2e12]            # FLOPs
    WIDTHS  = [16, 32, 64]
else:
    BUDGETS = [3e11, 1.2e12, 5e12]
    WIDTHS  = [16, 24, 32, 48, 64, 96]

n_layer   = 2
block_size, batch_size = 64, 32
lr = 1e-3
eval_iters = 40
tokens_per_step = batch_size * block_size

# ---------------------------------------------------------------------------
# SECTION 1 — Data (same tiny-Shakespeare pipeline as the other scripts)
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
# SECTION 2 — Model (identical GPT to nano_scaling_laws.py)
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

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.hs).transpose(1, 2)
                   for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        return x + self.ff(self.ln2(x))

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
    total = sum(p.numel() for p in m.parameters())
    return total - m.tok_emb.weight.numel() - m.pos_emb.weight.numel()

@torch.no_grad()
def val_loss(model):
    model.eval()
    ls = torch.zeros(eval_iters)
    for k in range(eval_iters):
        _, l = model(*get_batch("val"))
        ls[k] = l.item()
    model.train()
    return ls.mean().item()

# ---------------------------------------------------------------------------
# SECTION 3 — Run the IsoFLOP grid
# ---------------------------------------------------------------------------
# Chinchilla's key methodological fix over Kaplan: the LR schedule must be
# matched to the actual training horizon of each run (their Appendix B —
# "cosine cycle length should decay ~10x over approximately D tokens").
# Kaplan reused one schedule for every horizon, which systematically
# penalised the long-data runs and inflated the model-size exponent to 0.73.
# We honour the fix with a cosine schedule stretched to each run's length.
# ---------------------------------------------------------------------------
runs = []
for C in BUDGETS:
    for width in WIDTHS:
        n_head = min(4, max(1, width // 16))
        model = GPT(width, n_head).to(device)
        N = non_embedding_params(model)
        D = C / (6 * N)                              # token allowance
        steps = max(20, int(D / tokens_per_step))    # derived, not chosen
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=steps, eta_min=lr / 10)       # the Chinchilla fix
        t0 = time.time()
        for it in range(steps):
            xb, yb = get_batch("train")
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            opt.step(); sched.step()
        L = val_loss(model)
        runs.append({"C": C, "N": N, "D": D, "steps": steps, "loss": L,
                     "tokens_per_param": D / N})
        print(f"C={C:.1e} | width {width:3d} | N={N:8,} | D={D:9.2e} "
              f"| steps {steps:5d} | loss {L:.4f} | {time.time()-t0:5.1f}s")

# ---------------------------------------------------------------------------
# SECTION 4 — Approach 2: parabola fit per budget -> N_opt(C) -> exponent a
# ---------------------------------------------------------------------------
# The paper fits a parabola to each IsoFLOP curve in log-N space and reads
# the minimum. Then a straight line through (log C, log N_opt) gives a.
# ---------------------------------------------------------------------------
valleys = []
for C in BUDGETS:
    pts = [(r["N"], r["loss"]) for r in runs if r["C"] == C]
    x = np.log10([p_[0] for p_ in pts]); y = np.array([p_[1] for p_ in pts])
    coef = np.polyfit(x, y, 2)                       # parabola in log10 N
    if coef[0] > 0:                                  # opens upward -> valley
        logN_opt = -coef[1] / (2 * coef[0])
        # clamp inside tested range: extrapolated minima are meaningless
        logN_opt = float(np.clip(logN_opt, x.min(), x.max()))
    else:                                            # no interior valley seen
        logN_opt = float(x[np.argmin(y)])
    N_opt = 10 ** logN_opt
    valleys.append({"C": C, "N_opt": N_opt, "D_opt": C / (6 * N_opt),
                    "tokens_per_param": C / (6 * N_opt ** 2)})
    print(f"budget C={C:.1e}: valley at N_opt≈{N_opt:,.0f} "
          f"({valleys[-1]['tokens_per_param']:.0f} tokens/param)")

a_fit = b_fit = float("nan")
if len(valleys) >= 2:
    lc = np.log10([v["C"] for v in valleys]); ln = np.log10([v["N_opt"] for v in valleys])
    a_fit = float(np.polyfit(lc, ln, 1)[0])
    b_fit = 1 - a_fit                                # since C=6ND => b=1-a
    print(f"\nApproach 2 fit:  N_opt ∝ C^{a_fit:.2f}   D_opt ∝ C^{b_fit:.2f}")
    print("  Chinchilla: a≈0.50, b≈0.50   |   Kaplan: a=0.73, b=0.27")

# ---------------------------------------------------------------------------
# SECTION 5 — Approach 3: parametric fit  L(N,D) = E + A/N^α + B/D^β
# ---------------------------------------------------------------------------
# We minimise a Huber loss on log L, as the paper does (their Eq. 11, with
# LBFGS + a grid of initialisations). From (α, β) the closed-form frontier
# exponents follow:  a = β/(α+β),  b = α/(α+β)  (their Eq. 4).
# ---------------------------------------------------------------------------
def fit_parametric(runs):
    Ns = torch.tensor([r["N"] for r in runs], dtype=torch.float64)
    Ds = torch.tensor([r["D"] for r in runs], dtype=torch.float64)
    Ls = torch.tensor([r["loss"] for r in runs], dtype=torch.float64)
    best = None
    for a0 in (0.5, 2.0):                 # grid of initialisations (paper: 5x5)
        for b0 in (0.5, 2.0):
            # parameters in log space keeps A,B,E positive
            th = torch.tensor([a0, b0, 0.3, 0.3, math.log(1.0)],
                              dtype=torch.float64, requires_grad=True)
            opt = torch.optim.LBFGS([th], max_iter=200, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                la, lb, alpha, beta, le = th
                pred = torch.logsumexp(torch.stack([
                    la - alpha * torch.log(Ns),
                    lb - beta * torch.log(Ds),
                    le.expand_as(Ns)]), dim=0)       # log(A/N^α + B/D^β + E)
                err = pred - torch.log(Ls)
                loss = F.huber_loss(err, torch.zeros_like(err), delta=1e-3)
                loss.backward(); return loss
            try:
                fin = opt.step(closure).item()
            except Exception:
                continue
            if best is None or fin < best[0]:
                best = (fin, th.detach().clone())
    if best is None:
        return None
    la, lb, alpha, beta, le = best[1].tolist()
    return {"E": math.exp(le), "A": math.exp(la), "B": math.exp(lb),
            "alpha": alpha, "beta": beta,
            "a_frontier": beta / (alpha + beta),
            "b_frontier": alpha / (alpha + beta)}

par = fit_parametric(runs)
if par:
    print(f"\nApproach 3 fit:  L(N,D) = {par['E']:.2f} + {par['A']:.1f}/N^{par['alpha']:.2f}"
          f" + {par['B']:.1f}/D^{par['beta']:.2f}")
    print(f"  frontier exponents: a={par['a_frontier']:.2f}, b={par['b_frontier']:.2f}")
    print("  (paper on MassiveText: E=1.69, A=406.4, B=410.7, α=0.34, β=0.28"
          " -> a=0.46, b=0.54)")

# ---------------------------------------------------------------------------
# SECTION 6 — Plot the IsoFLOP valleys + save everything
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#5B8DEF", "#F5A524", "#5BE3A6", "#B98CFF"]
    for i, C in enumerate(BUDGETS):
        pts = sorted([(r["N"], r["loss"]) for r in runs if r["C"] == C])
        ax.semilogx([p_[0] for p_ in pts], [p_[1] for p_ in pts], "o-",
                    color=colors[i % 4], label=f"C={C:.0e} FLOPs")
        v = valleys[i]
        ax.axvline(v["N_opt"], color=colors[i % 4], ls=":", alpha=0.6)
    ax.set_xlabel("N — non-embedding parameters")
    ax.set_ylabel("final val loss (nats/char)")
    ax.set_title("nano-Chinchilla IsoFLOP profiles — each budget has a valley")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "runs", "chinchilla_isoflop.png"), dpi=150)
    print("wrote runs/chinchilla_isoflop.png")
except Exception as e:
    print("plot skipped:", e)

json.dump({
    "name": "chinchilla_isoflop",
    "runs": runs, "valleys": valleys,
    "approach2_fit": {"a": a_fit, "b": b_fit,
                      "chinchilla_says": {"a": 0.50, "b": 0.50},
                      "kaplan_says": {"a": 0.73, "b": 0.27}},
    "approach3_parametric": par,
    "verdict_hint": "Pass/fail question: is the fitted a closer to 0.5 "
                    "(Chinchilla) than to 0.73 (Kaplan)? Valleys visible in "
                    "every IsoFLOP curve = the paper's core phenomenon.",
}, open(os.path.join(HERE, "runs", "chinchilla.json"), "w"), indent=2)
print("wrote runs/chinchilla.json")
