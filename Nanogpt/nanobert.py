"""
=============================================================================
 nanobert.py — the SAME Transformer, with the causal mask deleted
=============================================================================
 Paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language
 Understanding" (Devlin et al., 2018).

 GPT vs BERT IN ONE SENTENCE
 ---------------------------
 GPT reads left-to-right and predicts the NEXT token (so it can generate);
 BERT reads in BOTH directions at once and predicts tokens that have been
 MASKED OUT (so it can understand, but not generate).

 Concretely, only three things change relative to nanogpt_attention.py:
   1. The causal mask is REMOVED — every position attends to every other.
   2. The objective becomes MASKED LANGUAGE MODELLING (MLM): hide 15% of
      the input, ask the model to reconstruct it, score ONLY those spots.
   3. Generation is replaced by fill-in-the-blank inference.
 Everything else (heads, blocks, residuals, LayerNorm) is identical —
 run a diff against nanogpt_attention.py and you'll see it.

 (The real BERT also has a Next Sentence Prediction task, WordPiece
 tokens and [CLS]/[SEP] machinery for fine-tuning; at character level on
 Shakespeare those add nothing to understanding the mechanism, so this
 file implements the MLM heart of BERT. The paper's own ablation — and
 RoBERTa a year later — found NSP contributes little.)

 RUN IT
 ------
   python nanobert.py            # full run (~15-25 min on CPU)
   python nanobert.py --quick    # smoke test (~2 min)

 Writes runs/bert_mlm.json + runs/bert_mlm.pt, then demos fill-in-the-blank.

 IMPORTANT FOR COMPARISON (see compare.py):
 BERT's MLM loss and GPT's next-token loss are NOT directly comparable —
 BERT predicts 15% of characters seeing both sides; GPT predicts 100% of
 characters seeing only the left. Expect BERT's number to be lower; that
 does not mean it is "better", it means its exam is easier per question.
=============================================================================
"""

import argparse, json, os, sys, time, urllib.request
import torch

# Windows consoles default to cp1252, which can't print the [MASK] glyph.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# SECTION 0 — Configuration (mirrors nanogpt_attention.py for fair comparison)
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
args = p.parse_args()

torch.manual_seed(1337)
device     = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 128 if not args.quick else 64
batch_size = 32  if not args.quick else 16
n_embd     = 128 if not args.quick else 64
n_head     = 4
n_layer    = 4  if not args.quick else 2
dropout    = 0.1
max_iters  = 3000 if not args.quick else 300
eval_every = 250  if not args.quick else 100
eval_iters = 100  if not args.quick else 20
lr         = 3e-4

# --- BERT's three magic numbers (Section 3.1 of the paper) ---
MASK_FRAC    = 0.15   # 15% of positions are selected for prediction
KEEP_MASK    = 0.80   # …of those: 80% replaced by [MASK]
KEEP_RANDOM  = 0.10   # …10% replaced by a RANDOM character
                      # …remaining 10% left UNCHANGED.
# Why 80/10/10 and not 100% [MASK]? Because [MASK] never appears at
# fine-tuning/inference time. If the model only ever predicted at [MASK]
# positions it would learn "only masked slots matter". Randoms + unchanged
# force it to keep a good representation of EVERY position.

# ---------------------------------------------------------------------------
# SECTION 1 — Data: same tiny-Shakespeare, plus a [MASK] token
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "input.txt")
if not os.path.exists(DATA):
    print("downloading tiny-shakespeare …")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt", DATA)
text = open(DATA, encoding="utf-8").read()

chars = sorted(set(text))
MASK_ID = len(chars)                    # one extra vocab slot for [MASK]
vocab_size = len(chars) + 1
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
itos[MASK_ID] = "▮"                     # display glyph for [MASK]
encode = lambda s: [stoi[c] for c in s]
decode = lambda t: "".join(itos[i] for i in t)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def mask_batch(x):
    """Apply the BERT corruption recipe to a clean batch x.
    Returns (corrupted_input, labels) where labels = original character at
    selected positions and -100 (PyTorch's ignore_index) everywhere else —
    so the loss is computed ONLY where the model had to reconstruct."""
    labels = torch.full_like(x, -100)
    selected = torch.rand_like(x, dtype=torch.float) < MASK_FRAC
    labels[selected] = x[selected]

    corrupted = x.clone()
    r = torch.rand_like(x, dtype=torch.float)
    corrupted[selected & (r < KEEP_MASK)] = MASK_ID                    # 80% -> [MASK]
    rand_slot = selected & (r >= KEEP_MASK) & (r < KEEP_MASK + KEEP_RANDOM)
    corrupted[rand_slot] = torch.randint(len(chars), x.shape)[rand_slot]  # 10% -> random
    # final 10%: left as the true character (but still predicted)
    return corrupted, labels

def get_batch(split):
    d  = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i:i + block_size] for i in ix])
    xc, y = mask_batch(x)
    return xc.to(device), y.to(device)

# ---------------------------------------------------------------------------
# SECTION 2 — Bidirectional self-attention: ONE line different from GPT
# ---------------------------------------------------------------------------
# Compare with nanogpt_attention.py's Head: the tril buffer and the
# masked_fill line are GONE. That's it. Every position now attends to all
# T positions — "deeply bidirectional" is the deletion of a mask.
# The price: you can no longer generate left-to-right, because position t
# was trained expecting to see t+1, t+2, …
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k = self.key(x); q = self.query(x)
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        att = F.softmax(att, dim=-1)          # NO causal mask — bidirectional
        return self.dropout(att) @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList(Head(hs) for _ in range(n_head))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], -1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # BERT uses GELU (a smooth ReLU) — we keep that detail from the paper.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa, self.ff = MultiHeadAttention(), FeedForward()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ---------------------------------------------------------------------------
# SECTION 3 — The BERT encoder + MLM head
# ---------------------------------------------------------------------------
# Same embeddings-blocks-head shape as GPT. The "MLM head" here is the same
# Linear-to-vocab projection; the difference lives entirely in (a) no causal
# mask and (b) which positions the loss looks at (ignore_index=-100 does it).
# ---------------------------------------------------------------------------
class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.mlm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, labels=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        logits = self.mlm_head(self.ln_f(self.blocks(x)))
        loss = None
        if labels is not None:
            # ignore_index=-100: unmasked positions contribute NOTHING.
            loss = F.cross_entropy(logits.view(B * T, -1), labels.view(B * T),
                                   ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def fill_in(self, s, mask_positions):
        """BERT's party trick: give it text with holes, it fills the holes
        using context from BOTH sides simultaneously."""
        ids = torch.tensor([encode(s)], device=device)
        for pos in mask_positions:
            ids[0, pos] = MASK_ID
        shown = decode(ids[0].tolist())
        logits, _ = self(ids)
        for pos in mask_positions:
            ids[0, pos] = logits[0, pos].argmax()
        return shown, decode(ids[0].tolist())

# ---------------------------------------------------------------------------
# SECTION 4 — Training loop (identical harness to the GPT script)
# ---------------------------------------------------------------------------
model = BERT().to(device)
n_params = sum(p.numel() for p in model.parameters())
n_nonemb = n_params - model.tok_emb.weight.numel() - model.pos_emb.weight.numel()
print(f"device={device}  params={n_params:,}  non-embedding={n_nonemb:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

@torch.no_grad()
def estimate_loss():
    model.eval(); out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(); return out

history = {"steps": [], "train": [], "val": []}
t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_every == 0:
        L = estimate_loss()
        history["steps"].append(it); history["train"].append(L["train"])
        history["val"].append(L["val"])
        print(f"step {it:5d} | train {L['train']:.4f} | val {L['val']:.4f} "
              f"| {time.time()-t0:6.1f}s")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------------------------------------------------------------------
# SECTION 5 — Save + fill-in-the-blank demo
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
result = {
    "name": "bert_mlm", "objective": "masked LM (15%, 80/10/10)",
    "params": n_params, "non_embedding_params": n_nonemb,
    "block_size": block_size, "n_layer": n_layer, "n_embd": n_embd,
    "final_val_loss_nats": history["val"][-1], "history": history,
    "train_flops_6NBS": 6 * n_nonemb * batch_size * block_size * max_iters,
    "loss_caveat": "MLM loss scores only ~15% masked positions with "
                   "bidirectional context — not comparable 1:1 with GPT's "
                   "next-token loss.",
}
json.dump(result, open(os.path.join(HERE, "runs", "bert_mlm.json"), "w"), indent=2)
torch.save(model.state_dict(), os.path.join(HERE, "runs", "bert_mlm.pt"))

print("\n----- fill-in-the-blank demo -----")
model.eval()
phrase = "O Romeo, Romeo! wherefore art thou Romeo?"
for positions in ([2, 3, 4], [10, 11, 12, 13], [36, 37, 38, 39, 40]):
    shown, filled = model.fill_in(phrase, positions)
    print(f"in : {shown}\nout: {filled}\n")
print("wrote runs/bert_mlm.json + .pt")
