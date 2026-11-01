# %% [markdown]
# # GPT-2 (124M) architecture lab: train it, then benchmark it
#
# Train a 124M-parameter GPT-2 from scratch on FineWeb, then train the same-size model with the
# modern changes from the LLM Inference Field Guide (RoPE, RMSNorm, SwiGLU, QK-norm, GQA), and compare
# them on one benchmark. Same data, same token budget, same schedule: only the architecture changes.
#
# **How to use it on Colab**
# 1. Runtime → Change runtime type → **A100** or **L4** GPU.
# 2. Edit the **Config** cell: which variants (`RUNS`) and how long (`BUDGET`).
# 3. Runtime → Run all. Checkpoints and results go to Google Drive, so a disconnected session resumes
#    where it stopped when you run all again, and runs from different sessions end up in one comparison.
#
# **The benchmark** (computed for every run, and for OpenAI's released GPT-2 124M as a reference)
# | Metric | What it tells you |
# |---|---|
# | Validation loss (and perplexity) on 10M held-out FineWeb tokens | How well it models text. The main score; lower is better |
# | Val loss vs tokens and vs wall-clock time | Sample efficiency, and whether a change pays for its own cost |
# | HellaSwag accuracy (10,042 questions, zero-shot) | Common-sense completion. Random = 25%; OpenAI GPT-2 124M ≈ 29.5% |
# | Training tokens/s, MFU, peak memory | Training cost on your GPU |
# | KV-cache bytes per token, prefill and decode tokens/s (batch 1 and 16) | Inference cost: where GQA/MQA pay off |
#
# Time per run is roughly *tokens ÷ throughput*: about 120–170k tokens/s on an A100 and 35–55k on an L4
# (see the README for a table).

# %%
# Setup: imports, GPU detection, paths.
import os, sys, math, time, json, glob, gc, subprocess
from dataclasses import dataclass, asdict, replace

IN_COLAB = 'google.colab' in sys.modules
def pip(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=True)
try:
    import tiktoken
except ImportError:
    pip('tiktoken'); import tiktoken
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'
NATIVE_BF16 = dev == 'cuda' and torch.cuda.get_device_capability()[0] >= 8  # Ampere+ (A100, L4); T4 is not
DTYPE = torch.bfloat16 if NATIVE_BF16 else torch.float16 if dev == 'cuda' else torch.float32
# Dense 16-bit tensor-core peak, for MFU (model FLOPs utilization).
PEAKS = {'A100': 312e12, 'H100': 989e12, 'L4': 121e12, 'A10': 125e12, 'T4': 65e12, 'V100': 125e12}
PEAK = next((v for k, v in PEAKS.items() if k in GPU), None)
# Memory bandwidth, for arithmetic intensity and the roofline (A100: 40 GB card 1.56 TB/s, 80 GB card 2.04 TB/s).
BWS = {'A100': 2039e9 if dev == 'cuda' and torch.cuda.get_device_properties(0).total_memory > 60e9 else 1555e9,
       'H100': 3350e9, 'L4': 300e9, 'A10': 600e9, 'T4': 320e9, 'V100': 900e9}
BW = next((v for k, v in BWS.items() if k in GPU), None)
RIDGE = PEAK / BW if PEAK and BW else None   # FLOPs per byte where a kernel stops being memory-bound
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
def autocast():                                   # mixed precision on GPU; plain fp32 on CPU (bf16 is emulated there)
    return torch.autocast(device_type=dev, dtype=DTYPE, enabled=dev == 'cuda')
def sync():
    if dev == 'cuda': torch.cuda.synchronize()
print(f'{GPU} · {DTYPE} · torch {torch.__version__}' + (f' · {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB' if dev == 'cuda' else ''))

# %%
# Config: the only cell you normally edit.
RUNS = ['gpt2', 'qwen3-style']   # any keys of VARIANTS below. Architecture family: gpt2, gpt2+rope, gpt2+rmsnorm,
                                 # gpt2+swiglu, gpt2+relu2, llama-style, qwen3-style, qwen3-mqa. Attention family
                                 # (llama-style base): llama-style (MHA), attn-gqa, attn-mqa, attn-mla, attn-swa,
                                 # attn-swa-global, attn-dsa, attn-gdn, attn-gdn-hybrid
BUDGET = 'standard'              # 'smoke' | 'quick' | 'standard' | 'long' (see BUDGETS)
LR = 1e-3                        # peak LR, same for every variant; other values get their own results (LR sweeps)
WEIGHT_DECAY = 0.1
SEED = 1337                      # same init seed and same data order for every run
COMPILE = True                   # torch.compile: ~1 min to compile, then ~1.5-2x faster
SAVE_TO_DRIVE = True             # Colab: keep checkpoints + results on Google Drive
HELLASWAG_N = None               # None = all 10,042 questions; a smaller number is faster but noisier
SHOTS = (0, 1, 5)                # HellaSwag zero-, one- and few-shot: solved examples shown before each question
FEWSHOT_N = 2000                 # questions used for the 1- and 5-shot runs (prompts are longer; ±1 point)
MODEL_BASE = {}                  # {} = GPT-2 124M. e.g. dict(n_layer=24, n_head=16, n_embd=1024) ≈ 350M

BUDGETS = {                      # tokens trained, tokens per optimizer step
    'smoke':    dict(tokens=20_000_000,    batch=131_072),   # pipeline check, a few minutes
    'quick':    dict(tokens=100_000_000,   batch=262_144),   # LR sweeps, rough comparisons
    'standard': dict(tokens=500_000_000,   batch=262_144),   # the comparison runs
    'long':     dict(tokens=2_500_000_000, batch=524_288),   # ~20 tokens/param (Chinchilla-optimal)
}

# %%
# Paths and data. FineWeb is downloaded already tokenized with GPT-2's tokenizer
# (100M tokens per 200 MB shard), so Colab doesn't spend an hour tokenizing.
if IN_COLAB and SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT = '/content/drive/MyDrive/gpt2-lab'
else:
    OUT = os.path.abspath('gpt2-lab-out')
DATA = '/content/fineweb10B' if IN_COLAB else os.path.abspath('fineweb10B')
for d in (OUT + '/ckpt', OUT + '/results', DATA):
    os.makedirs(d, exist_ok=True)

DEFAULT_LR = 1e-3
def run_tag(name, budget=None):
    """File name for a run: variant__budget, plus the LR when it isn't the default (so LR sweeps don't overwrite)."""
    t = f'{name}__{budget or BUDGET}'
    return t if LR == DEFAULT_LR else f'{t}__lr{LR:g}'

def get_data(train_tokens):
    from huggingface_hub import hf_hub_download
    n = min(103, math.ceil(train_tokens * 1.01 / 100e6))      # 1% spare: shard tails are skipped
    files = ['fineweb_val_000000.bin'] + [f'fineweb_train_{i:06d}.bin' for i in range(1, n + 1)]
    for f in files:
        if not os.path.exists(os.path.join(DATA, f)):
            hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename=f, repo_type='dataset', local_dir=DATA)
    return sorted(glob.glob(os.path.join(DATA, 'fineweb_train_*.bin')))[:n], os.path.join(DATA, 'fineweb_val_000000.bin')

def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, f'bad shard header in {path}'
    return np.memmap(path, dtype=np.uint16, mode='r', offset=256 * 4, shape=(int(header[2]),))

class Loader:
    """Reads shards in order, so every variant sees exactly the same tokens in the same order."""
    def __init__(self, files, B, T, state=None):
        self.files, self.B, self.T = files, B, T
        self.shard, self.pos = (state['shard'], state['pos']) if state else (0, 0)
        self.tokens = load_shard(files[self.shard])
    def state(self):
        return dict(shard=self.shard, pos=self.pos)
    def next(self):
        B, T = self.B, self.T
        if self.pos + B * T + 1 > len(self.tokens):
            self.shard = (self.shard + 1) % len(self.files)
            self.tokens, self.pos = load_shard(self.files[self.shard]), 0
        buf = torch.from_numpy(self.tokens[self.pos:self.pos + B * T + 1].astype(np.int64))
        self.pos += B * T
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        if dev == 'cuda':
            return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)
        return x, y

# %%
# The model: GPT-2, with every modern change behind a switch, including the attention mechanism itself.
@dataclass
class Cfg:
    vocab_size: int = 50304      # 50,257 padded to a multiple of 64 (faster matmuls)
    block_size: int = 1024       # context length
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_kv_head: int = 0           # 0 = n_head (multi-head, MHA); 4 = grouped-query (GQA); 1 = multi-query (MQA)
    norm: str = 'layernorm'      # 'layernorm' (GPT-2) | 'rmsnorm' (Llama, Qwen)
    pos: str = 'learned'         # 'learned' absolute positions (GPT-2) | 'rope' rotary
    mlp: str = 'gelu'            # 'gelu' (GPT-2) | 'swiglu' (Llama, Qwen) | 'relu2'
    mlp_hidden: int = 0          # 0 = auto: 4·d for gelu/relu2, 8/3·d for swiglu (same parameters)
    qk_norm: bool = False        # RMSNorm on each head's q and k before RoPE (Qwen3)
    bias: bool = True            # GPT-2 has biases; modern models drop them
    rope_theta: float = 10000.0  # Qwen3 uses 1e6 for long contexts; 1e4 suits 1,024 tokens
    attn: str = 'mha'            # 'mha' (+ n_kv_head → GQA/MQA) | 'swa' | 'mla' | 'dsa' | 'gdn'
    hybrid_every: int = 0        # >0: every Nth layer is full attention, the others use `attn` (Gemma 3, Qwen3-Next)
    window: int = 256            # swa: each query sees only the last `window` tokens
    mla_rank: int = 256          # mla: size of the K/V latent cached per token
    mla_rope: int = 32           # mla: decoupled RoPE dims (a small key shared by all heads)
    dsa_topk: int = 256          # dsa: keys each query attends to
    dsa_heads: int = 4           # dsa: lightning-indexer heads
    dsa_dim: int = 32            # dsa: lightning-indexer dims
    gdn_heads: int = 6           # gdn: heads; each keeps a (n_embd/gdn_heads)² state matrix

def layer_kind(c, i):
    return 'mha' if c.hybrid_every and (i + 1) % c.hybrid_every == 0 else c.attn

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps).type_as(x)   # autocast may return fp32

def Norm(c, d):
    return RMSNorm(d) if c.norm == 'rmsnorm' else nn.LayerNorm(d, bias=c.bias)

def rope_tables(T, hd, theta):
    inv = 1.0 / theta ** (torch.arange(0, hd, 2).float() / hd)
    f = torch.outer(torch.arange(T).float(), inv)            # [T, hd/2]
    return f.cos(), f.sin()

def apply_rope(x, cos, sin):                                  # x: [B, H, T, hd]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)                               # "rotate half" layout, as in HF Llama/Qwen
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Rope(nn.Module):
    def __init__(self, T, dim, theta):
        super().__init__()
        cos, sin = rope_tables(T, dim, theta)
        self.register_buffer('cos', cos, persistent=False); self.register_buffer('sin', sin, persistent=False)
    def forward(self, x, pos):                                # x: [B, H, T, dim], first token at position pos
        T = x.size(2)
        return apply_rope(x, self.cos[pos:pos + T], self.sin[pos:pos + T])

def causal_mask(Tq, Tk, pos, window, device):
    """True where a query (absolute position pos+i) may see a key (position j): j ≤ i, and within `window` if set."""
    qi = torch.arange(Tq, device=device)[:, None] + pos
    kj = torch.arange(Tk, device=device)[None, :]
    m = kj <= qi
    return m & (qi - kj < window) if window else m

def attend(q, k, v, pos, window=0):
    """Causal softmax attention for T queries (positions pos..pos+T-1) over all S keys so far."""
    T, S = q.size(2), k.size(2)
    if T == 1:                                                # decoding one token: it may see every cached key
        if window and S > window:
            k, v = k[:, :, -window:], v[:, :, -window:]
        return F.scaled_dot_product_attention(q, k, v)
    if window and S > window:                                 # a band mask (dense kernel; real SWA kernels skip the zeros)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(T, S, pos, window, q.device))
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)    # FlashAttention on GPU

class KVCache:
    def __init__(self, B, n_kv, T, hd, dtype, device, extra_dim=0):
        self.k = torch.zeros(B, n_kv, T, hd, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
        self.extra = torch.zeros(B, T, extra_dim, dtype=dtype, device=device) if extra_dim else None
    def update(self, k, v, pos):
        T = k.size(2)
        self.k[:, :, pos:pos + T] = k; self.v[:, :, pos:pos + T] = v
        return self.k[:, :, :pos + T], self.v[:, :, :pos + T]
    def update_extra(self, e, pos):                           # DSA: the indexer's key for every token
        self.extra[:, pos:pos + e.size(1)] = e
        return self.extra[:, :pos + e.size(1)]

class LatentCache:                                            # MLA: one latent + one small RoPE key per token
    def __init__(self, B, T, rank, rope, dtype, device):
        self.c = torch.zeros(B, T, rank, dtype=dtype, device=device)
        self.kr = torch.zeros(B, 1, T, rope, dtype=dtype, device=device)
    def update(self, c, kr, pos):
        T = c.size(1)
        self.c[:, pos:pos + T] = c; self.kr[:, :, pos:pos + T] = kr
        return self.c[:, :pos + T], self.kr[:, :, :pos + T]

class StateCache:                                             # Gated DeltaNet: a fixed-size state, whatever the context
    def __init__(self):
        self.state, self.conv = None, None

class Attention(nn.Module):
    """Softmax attention. MHA: every head has its own K and V. GQA/MQA: groups of query heads share K/V heads
    (n_kv_head < n_head), shrinking the KV cache. Sliding window (window > 0): each query sees only recent tokens."""
    def __init__(self, c, window=0):
        super().__init__()
        self.nh, self.nkv, self.hd, self.window = c.n_head, c.n_kv_head or c.n_head, c.n_embd // c.n_head, window
        assert c.n_embd % self.nh == 0 and self.nh % self.nkv == 0
        self.q = nn.Linear(c.n_embd, self.nh * self.hd, bias=c.bias)
        self.kv = nn.Linear(c.n_embd, 2 * self.nkv * self.hd, bias=c.bias)   # GQA shrinks this
        self.o = nn.Linear(self.nh * self.hd, c.n_embd, bias=c.bias)
        self.o.residual = True
        self.qn = RMSNorm(self.hd) if c.qk_norm else None
        self.kn = RMSNorm(self.hd) if c.qk_norm else None
        self.rope = Rope(c.block_size, self.hd, c.rope_theta) if c.pos == 'rope' else None
    def qkv(self, x, pos):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd)
        k, v = self.kv(x).view(B, T, 2, self.nkv, self.hd).unbind(2)
        if self.qn is not None:
            q, k = self.qn(q), self.kn(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        return q, k, v
    def expand_kv(self, k, v):                                # share each K/V head across a group of query heads
        if self.nkv != self.nh:
            k = k.repeat_interleave(self.nh // self.nkv, dim=1)
            v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        return k, v
    def forward(self, x, cache=None, pos=0):
        B, T, _ = x.shape
        q, k, v = self.qkv(x, pos)
        if cache is not None:
            assert pos == 0 or T == 1, 'cache supports a prefill at pos 0, then one token at a time'
            k, v = cache.update(k, v, pos)
        k, v = self.expand_kv(k, v)
        y = attend(q, k, v, pos, self.window)
        return self.o(y.transpose(1, 2).reshape(B, T, -1)), None

class DSAttention(Attention):
    """DeepSeek Sparse Attention (V3.2). A small 'lightning indexer' scores every earlier token for each query,
    I(t,s) = Σ_j w_j(t) · ReLU(q_j(t) · k(s)), and the query attends only to its top-k keys. The indexer is trained
    to imitate the full attention pattern (a KL loss on its own detached input), after a short dense warm-up."""
    def __init__(self, c):
        super().__init__(c)
        self.topk, self.hi, self.di = c.dsa_topk, c.dsa_heads, c.dsa_dim
        self.iq = nn.Linear(c.n_embd, self.hi * self.di, bias=False)
        self.ik = nn.Linear(c.n_embd, self.di, bias=False)
        self.iw = nn.Linear(c.n_embd, self.hi, bias=False)
        self.dense = False                                    # True during the dense warm-up at the start of training
    def forward(self, x, cache=None, pos=0):
        B, T, _ = x.shape
        q, k, v = self.qkv(x, pos)
        xi = x.detach()                                       # the indexer is trained only by its own KL loss
        ik = self.ik(xi)                                      # [B, T, di]: one small indexer key per token
        if cache is not None:
            k, v = cache.update(k, v, pos); ik = cache.update_extra(ik, pos)
        S = k.size(2)
        iq = self.iq(xi).view(B, T, self.hi, self.di)
        score = torch.einsum('bthd,bsd->bths', iq, ik).relu() * self.di ** -0.5
        score = torch.einsum('bths,bth->bts', score, self.iw(xi) * self.hi ** -0.5)   # [B, T, S]
        allowed = causal_mask(T, S, pos, 0, x.device)
        if self.dense or S <= self.topk:
            mask = allowed.expand(B, T, S)
        else:
            recency = torch.arange(S, device=x.device) * 1e-6     # ReLU makes many scores exactly 0: break ties
            top = (score.float() + recency).masked_fill(~allowed, float('-inf')).topk(self.topk, dim=-1).indices   # by recency
            mask = torch.zeros(B, T, S, dtype=torch.bool, device=x.device).scatter_(-1, top, True) & allowed
        k2, v2 = self.expand_kv(k, v)
        y = F.scaled_dot_product_attention(q, k2, v2, attn_mask=mask[:, None])   # dense kernel + mask (no sparse kernel)
        aux = None
        if self.training:
            with torch.no_grad():                             # target: the full attention pattern, averaged over heads
                p = (q @ k2.transpose(-1, -2)).float().mul(self.hd ** -0.5).masked_fill(~allowed, float('-inf'))
                p = p.softmax(-1).mean(1)                     # [B, T, S]
            logp = score.float().masked_fill(~allowed, float('-inf')).log_softmax(-1).masked_fill(~allowed, 0.0)
            aux = F.kl_div(logp.reshape(-1, S), p.reshape(-1, S), reduction='batchmean')
        return self.o(y.transpose(1, 2).reshape(B, T, -1)), aux

class MLAttention(nn.Module):
    """Multi-head latent attention (DeepSeek-V2/V3). Each token is compressed into one small latent vector c
    (mla_rank numbers); every head's key and value are rebuilt from it by up-projections. The cache stores c plus a
    small RoPE key shared by all heads: mla_rank + mla_rope numbers per token per layer instead of 2 × heads × head_dim.
    RoPE sits in separate 'decoupled' dims because a position-dependent rotation can't be folded into the up-projection.
    (Inference engines go further and absorb the up-projections into q and o, so attention runs on the latent directly.)"""
    def __init__(self, c):
        super().__init__()
        d = c.n_embd
        self.nh, self.dn, self.dr, self.dv, self.r = c.n_head, d // c.n_head, c.mla_rope, d // c.n_head, c.mla_rank
        self.q = nn.Linear(d, self.nh * (self.dn + self.dr), bias=False)
        self.dkv = nn.Linear(d, self.r + self.dr, bias=False)    # down-projection: latent + shared RoPE key
        self.kv_norm = RMSNorm(self.r)
        self.uk = nn.Linear(self.r, self.nh * self.dn, bias=False)   # up-projections, per head
        self.uv = nn.Linear(self.r, self.nh * self.dv, bias=False)
        self.o = nn.Linear(self.nh * self.dv, d, bias=c.bias)
        self.o.residual = True
        self.rope = Rope(c.block_size, self.dr, c.rope_theta)
    def forward(self, x, cache=None, pos=0):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.dn + self.dr).transpose(1, 2)
        qn, qr = q.split([self.dn, self.dr], dim=-1)
        c, kr = self.dkv(x).split([self.r, self.dr], dim=-1)
        c = self.kv_norm(c)                                   # [B, T, r]: this is what gets cached
        kr = self.rope(kr.view(B, T, 1, self.dr).transpose(1, 2), pos)   # [B, 1, T, dr]
        if cache is not None:
            assert pos == 0 or T == 1
            c, kr = cache.update(c, kr, pos)
        S = c.size(1)
        kn = self.uk(c).view(B, S, self.nh, self.dn).transpose(1, 2)
        v = self.uv(c).view(B, S, self.nh, self.dv).transpose(1, 2)
        k = torch.cat([kn, kr.expand(B, self.nh, S, self.dr)], dim=-1)
        q = torch.cat([qn, self.rope(qr, pos)], dim=-1)
        y = attend(q, k, v, pos)
        return self.o(y.transpose(1, 2).reshape(B, T, -1)), None

def delta_rule_recurrent(q, k, v, g, beta, state=None):
    """The gated delta rule, one token at a time (same maths as fla-core's naive_recurrent_gated_delta_rule).
    q, k: [B, T, H, K] (unit length), v: [B, T, H, V], g: log-decay [B, T, H] (≤ 0), beta: [B, T, H] in (0, 1)."""
    B, T, H, K = q.shape
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    q = q * K ** -0.5
    S = q.new_zeros(B, H, K, v.size(-1)) if state is None else state.float()
    out = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]                     # forget: S ← α·S
        err = v[:, t] - torch.einsum('bhk,bhkv->bhv', k[:, t], S)  # what S currently returns for k, vs the new v
        S = S + k[:, t, :, :, None] * (beta[:, t, :, None] * err)[:, :, None, :]   # delta rule: S ← S + β·k·errᵀ
        out.append(torch.einsum('bhk,bhkv->bhv', q[:, t], S))      # read: o = Sᵀq
    return torch.stack(out, 1), S

FLA = None                                                    # fla-core's chunked Triton kernel, loaded on GPU
def load_fla():
    global FLA
    if FLA is not None or dev != 'cuda':
        return
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError:
        pip('fla-core'); from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    FLA = chunk_gated_delta_rule
    # self-check: the fast parallel kernel must match the token-by-token recurrence
    torch.manual_seed(0)
    B, T, H, K = 2, 100, 3, 32
    q, k = (F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1) for _ in range(2))
    v = torch.randn(B, T, H, K, device=dev); g = -F.softplus(torch.randn(B, T, H, device=dev)); beta = torch.rand(B, T, H, device=dev)
    o1, s1 = FLA(q, k, v, g, beta, scale=K ** -0.5, output_final_state=True)
    o2, s2 = delta_rule_recurrent(q, k, v, g, beta)
    err = max((o1.float() - o2).abs().max().item(), (s1.float() - s2).abs().max().item())
    assert err < 2e-2, f'fla-core chunk kernel disagrees with the reference recurrence: {err}'
    print(f'fla-core Gated DeltaNet kernel loaded and checked against the recurrence (max diff {err:.1e})')

def delta_rule(q, k, v, g, beta, state=None):
    if FLA is not None and q.is_cuda and q.size(1) > 1:       # training / prefill: chunked parallel kernel
        o, S = FLA(q.contiguous(), k.contiguous(), v.to(q.dtype).contiguous(), g.float().contiguous(),
                   beta.to(q.dtype).contiguous(), scale=q.size(-1) ** -0.5,
                   initial_state=state, output_final_state=True)
        return o, S
    o, S = delta_rule_recurrent(q, k, v, g, beta, state)      # decode (one token) or CPU
    return o.to(v.dtype), S

class GatedDeltaNet(nn.Module):
    """Gated DeltaNet (Yang et al., 2024), the linear-attention layer of Qwen3-Next. No KV cache: each head keeps a
    fixed K×V state matrix S. Per token: S ← α·S (forget, α from the input), S ← S + β·k·(v − Sᵀk)ᵀ (write v at
    address k, correcting what S already returned there: the delta rule), o = Sᵀq (read). Memory and per-token cost
    stay constant however long the context. Training uses a chunked parallel form that computes the same thing."""
    def __init__(self, c):
        super().__init__()
        d = c.n_embd
        self.H, self.K = c.gdn_heads, d // c.gdn_heads
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False); self.v = nn.Linear(d, d, bias=False)
        self.conv = nn.Conv1d(3 * d, 3 * d, kernel_size=4, groups=3 * d, bias=False)   # short causal conv over time
        self.a = nn.Linear(d, self.H, bias=False)                 # forget gate α (per head, per token)
        self.b = nn.Linear(d, self.H, bias=False)                 # write strength β
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(1, 16)))
        dt = torch.exp(torch.rand(self.H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus (Mamba-2 init)
        self.z = nn.Linear(d, d, bias=False)                      # output gate
        self.onorm = RMSNorm(self.K)
        self.o = nn.Linear(d, d, bias=c.bias)
        self.o.residual = True
    def forward(self, x, cache=None, pos=0):
        B, T, d = x.shape
        qkv = torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)          # [B, T, 3d]
        prev = cache.conv if cache is not None and cache.conv is not None else qkv.new_zeros(B, 3, 3 * d)
        xc = torch.cat([prev.to(qkv.dtype), qkv], dim=1)                     # 3 earlier tokens + this chunk
        if cache is not None:
            cache.conv = xc[:, -3:]
        qkv = F.silu(F.conv1d(xc.transpose(1, 2), self.conv.weight, groups=3 * d)).transpose(1, 2)
        q, k, v = qkv.reshape(B, T, 3, self.H, self.K).unbind(2)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        beta = self.b(x).sigmoid()
        g = -self.A_log.exp() * F.softplus(self.a(x).float() + self.dt_bias)  # log α ≤ 0
        o, state = delta_rule(q, k, v, g, beta, cache.state if cache is not None else None)
        if cache is not None:
            cache.state = state
        o = self.onorm(o) * F.silu(self.z(x).view(B, T, self.H, self.K))      # gated output norm (Qwen3-Next)
        return self.o(o.reshape(B, T, d)), None

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.kind = c.mlp
        h = c.mlp_hidden or (int(8 / 3 * c.n_embd / 64 + 0.5) * 64 if c.mlp == 'swiglu' else 4 * c.n_embd)
        self.up = nn.Linear(c.n_embd, (2 if c.mlp == 'swiglu' else 1) * h, bias=c.bias)   # swiglu: gate and up fused
        self.down = nn.Linear(h, c.n_embd, bias=c.bias)
        self.down.residual = True
    def forward(self, x):
        if self.kind == 'swiglu':
            g, u = self.up(x).chunk(2, dim=-1)
            return self.down(F.silu(g) * u)
        h = self.up(x)
        return self.down(F.gelu(h, approximate='tanh') if self.kind == 'gelu' else F.relu(h).square())

ATTENTION = {'mha': lambda c: Attention(c), 'swa': lambda c: Attention(c, c.window), 'mla': MLAttention,
             'dsa': DSAttention, 'gdn': GatedDeltaNet}

class Block(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.kind = layer_kind(c, i)
        self.n1, self.attn, self.n2, self.mlp = Norm(c, c.n_embd), ATTENTION[self.kind](c), Norm(c, c.n_embd), MLP(c)
    def forward(self, x, cache=None, pos=0):
        a, aux = self.attn(self.n1(x), cache, pos)            # pre-norm residual blocks
        x = x + a
        return x + self.mlp(self.n2(x)), aux

class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd) if c.pos == 'learned' else None
        self.blocks = nn.ModuleList(Block(c, i) for i in range(c.n_layer))
        self.nf = Norm(c, c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.wte.weight                    # tied input/output embeddings
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.c.n_layer) if getattr(m, 'residual', False) else 0.02
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, idx, targets=None, caches=None, pos=0, last_only=False):
        B, T = idx.shape
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(pos, pos + T, device=idx.device))
        aux = 0.0
        for i, blk in enumerate(self.blocks):
            x, a = blk(x, caches[i] if caches else None, pos)
            if a is not None:
                aux = aux + a                                 # DSA's indexer loss (0 for every other attention)
        x = self.nf(x)
        if last_only:
            x = x[:, -1:]
        logits = self.head(x)
        loss = None if targets is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1)) + aux
        return logits, loss
    def set_dense_warmup(self, on):                           # DSA: attend densely while the indexer learns
        for m in self.modules():
            if isinstance(m, DSAttention):
                m.dense = on
    def new_caches(self, B, T):
        d = self.wte.weight.device; out = []
        for blk in self.blocks:
            a = blk.attn
            if isinstance(a, GatedDeltaNet):
                out.append(StateCache())
            elif isinstance(a, MLAttention):
                out.append(LatentCache(B, T, a.r, a.dr, DTYPE, d))
            else:
                out.append(KVCache(B, a.nkv, T, a.hd, DTYPE, d, a.di if isinstance(a, DSAttention) else 0))
        return out
    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())         # tied weights counted once
        if non_embedding:
            n -= self.wte.weight.numel() + (self.wpe.weight.numel() if self.wpe is not None else 0)
        return n
    def cache_bytes(self, S, bytes_per_value=2):
        """Inference cache for ONE sequence at context length S, summed over layers (16-bit values)."""
        total = 0
        for blk in self.blocks:
            a = blk.attn
            if isinstance(a, GatedDeltaNet):
                total += a.H * a.K * a.K                      # fixed state, independent of S
            elif isinstance(a, MLAttention):
                total += S * (a.r + a.dr)
            else:
                total += (min(S, a.window) if a.window else S) * 2 * a.nkv * a.hd
                if isinstance(a, DSAttention):
                    total += S * a.di                         # the indexer's keys
        return total * bytes_per_value
    def kv_bytes_per_token(self, bytes_per_value=2):          # average per token over a 1,024-token context
        return self.cache_bytes(1024, bytes_per_value) / 1024
    def step_cost(self, B, S, T=1, bytes_per_value=2, absorbed=True):
        """FLOPs and HBM bytes of one forward step: B sequences each add T new tokens after S cached ones.
        FLOPs: 2 per weight per token (all matmuls, incl. the tied lm_head) + the attention itself, per mechanism.
        Bytes: every weight read once + whatever cache this mechanism reads (16-bit values).
        absorbed: MLA as inference engines run it (attention directly on the cached latent)."""
        mm = sum(m.weight.numel() for m in self.modules() if isinstance(m, nn.Linear))
        flops, bytes_ = 2 * mm * B * T, self.n_params() * bytes_per_value
        keys = S + (T + 1) / 2                                # average keys each query attends to (causal)
        for blk in self.blocks:
            a = blk.attn
            if isinstance(a, GatedDeltaNet):                  # decay, read, correct, write, query: ~7 ops per state entry
                flops += 7 * B * T * a.H * a.K * a.K
                bytes_ += B * a.H * a.K * a.K * bytes_per_value
            elif isinstance(a, MLAttention):
                qk, vd = (a.r + a.dr, a.r) if absorbed else (a.dn + a.dr, a.dv)
                flops += 2 * B * T * keys * a.nh * (qk + vd)
                bytes_ += B * (S + T) * (a.r + a.dr) * bytes_per_value
            else:
                kv_tok = 2 * a.nkv * a.hd * bytes_per_value
                read = S + T                                  # keys read from memory (prefill reads them all once)
                seen = keys
                if a.window:
                    seen = min(keys, a.window); read = min(S + T, a.window) if T == 1 else S + T
                if isinstance(a, DSAttention):
                    seen = min(keys, a.topk); read = min(S + T, a.topk) if T == 1 else S + T
                    flops += 2 * B * T * keys * a.hi * a.di   # the indexer scores every key
                    bytes_ += B * (S + T) * a.di * bytes_per_value
                flops += 2 * B * T * seen * a.nh * 2 * a.hd
                bytes_ += B * read * kv_tok
        return flops, bytes_
    def intensity(self, B, S, T=1):
        """Arithmetic intensity, FLOPs per byte moved. Below the GPU's ridge point the step is memory-bound."""
        f, b = self.step_cost(B, S, T)
        return f / b
    def flops_per_token(self):                                # training: forward + backward ≈ 3 × forward
        f, _ = self.step_cost(1, 0, self.c.block_size, absorbed=False)
        return 3 * f / self.c.block_size

# Inference scenarios for arithmetic intensity: (batch, cached tokens, new tokens per sequence)
AI_POINTS = {'decode b1': (1, 1024, 1), 'decode b16': (16, 1024, 1), 'decode b128': (128, 1024, 1), 'prefill 1K': (1, 0, 1024)}
def intensity_table(model):
    return {k: model.intensity(*v) for k, v in AI_POINTS.items()}

# %%
# The variants. Each ablation changes one thing from GPT-2, so you can see cause → effect.
VARIANTS = {
    'gpt2':         dict(),                                   # the 2019 recipe
    'gpt2+rope':    dict(pos='rope'),
    'gpt2+rmsnorm': dict(norm='rmsnorm'),
    'gpt2+swiglu':  dict(mlp='swiglu'),
    'gpt2+relu2':   dict(mlp='relu2'),
    'llama-style':  dict(pos='rope', norm='rmsnorm', mlp='swiglu', bias=False),
    'qwen3-style':  dict(pos='rope', norm='rmsnorm', mlp='swiglu', bias=False, qk_norm=True, n_kv_head=4, match_params=True),
    'qwen3-mqa':    dict(pos='rope', norm='rmsnorm', mlp='swiglu', bias=False, qk_norm=True, n_kv_head=1, match_params=True),
}
# The attention family: everything else fixed (the llama-style base: RoPE, RMSNorm, SwiGLU, no biases), only the
# attention mechanism changes. llama-style itself is the MHA reference.
ATTN_BASE = dict(pos='rope', norm='rmsnorm', mlp='swiglu', bias=False, match_params='llama-style')
VARIANTS.update({
    'attn-gqa':        dict(ATTN_BASE, n_kv_head=4),                 # 12 query heads share 4 K/V heads (Llama 3, Qwen)
    'attn-mqa':        dict(ATTN_BASE, n_kv_head=1),                 # all 12 share 1 K/V head (PaLM, Falcon)
    'attn-mla':        dict(ATTN_BASE, attn='mla'),                  # K/V from a small cached latent (DeepSeek V2/V3)
    'attn-swa':        dict(ATTN_BASE, attn='swa'),                  # every layer sees the last 256 tokens (Mistral 7B)
    'attn-swa-global': dict(ATTN_BASE, attn='swa', hybrid_every=6),  # 5 local : 1 global (Gemma 3)
    'attn-dsa':        dict(ATTN_BASE, attn='dsa'),                  # indexer picks top-256 keys (DeepSeek V3.2)
    'attn-gdn':        dict(ATTN_BASE, attn='gdn'),                  # Gated DeltaNet linear attention, fixed state
    'attn-gdn-hybrid': dict(ATTN_BASE, attn='gdn', hybrid_every=4),  # 3 GDN : 1 full attention (Qwen3-Next)
})
# match_params: a mechanism that removes (or adds) attention weights gets its MLP widened (or narrowed) to keep the
# same total as the reference (GPT-2, or the variant named), so a loss gap can't just mean "fewer parameters".

def count_params(c):
    with torch.device('meta'):
        return GPT(c).n_params()

def make_cfg(name):
    v = dict(VARIANTS[name]); match = v.pop('match_params', False)
    c = Cfg(**{**MODEL_BASE, **v})
    if match:
        target = count_params(make_cfg(match) if isinstance(match, str) else Cfg(**MODEL_BASE))
        best = min(range(64, 8 * c.n_embd, 64), key=lambda h: abs(count_params(replace(c, mlp_hidden=h)) - target))
        c = replace(c, mlp_hidden=best)
    return c

for name in VARIANTS:
    c = make_cfg(name)
    with torch.device('meta'):
        m = GPT(c)
    ai = intensity_table(m)
    print(f'{name:16s} {m.n_params()/1e6:6.1f}M params · cache @1K ctx {m.cache_bytes(1024)/2**20:5.2f} MiB/sequence · '
          f'intensity (FLOP/byte) decode b1/b16/b128 @1K ctx {ai["decode b1"]:4.1f} / {ai["decode b16"]:4.1f} / {ai["decode b128"]:5.1f}'
          f' · prefill {ai["prefill 1K"]:5.0f} · mlp_hidden {c.mlp_hidden or "auto"}')
if RIDGE:
    print(f'\n{GPU} ridge point: {RIDGE:.0f} FLOP/byte. Steps below it are memory-bound (decode), above it compute-bound (prefill).')

# %%
# The benchmark: validation loss, HellaSwag, training speed, inference speed.
enc = tiktoken.get_encoding('gpt2')

@torch.no_grad()
def val_loss(logits_fn, tokens, B, T=1024):
    """Mean next-token loss over a fixed slice of the held-out shard."""
    n = (len(tokens) - 1) // (B * T); total = 0.0
    for i in range(n):
        buf = tokens[i * B * T:(i + 1) * B * T + 1].to(dev)
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        with autocast():
            logits = logits_fn(x)
        total += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)).item()
    return total / n

def load_hellaswag(split='validation'):
    """validation = the 10,042 test questions; train = the pool few-shot examples are drawn from."""
    cache = os.path.join(DATA, f'hellaswag_{"val" if split == "validation" else split}.jsonl')
    if not os.path.exists(cache):
        # The original GitHub repo is DMCA-blocked (2026); the Hugging Face copy is used instead.
        import pandas as pd
        from huggingface_hub import hf_hub_download
        p = hf_hub_download('Rowan/hellaswag', f'data/{split}-00000-of-00001.parquet', repo_type='dataset', local_dir=DATA)
        with open(cache, 'w') as f:
            for r in pd.read_parquet(p).itertuples():
                f.write(json.dumps(dict(ctx=r.ctx, endings=list(r.endings), label=int(r.label))) + '\n')
    with open(cache) as f:
        return [json.loads(l) for l in f]

@torch.no_grad()
def hellaswag(logits_fn, examples, shots=0, demos=None, max_len=1024, per_batch=8):
    """Score each of the 4 endings by the model's loss on the ending tokens; pick the lowest.
    acc uses the summed loss, acc_norm the per-token average (the number usually reported for GPT-2).
    shots > 0: before each question, show `shots` solved examples (context + correct ending) from the train split,
    a different random draw per question, as lm-eval does. Only the answer tokens are ever scored."""
    import random
    right = right_norm = 0; correct_norm = []
    for i in range(0, len(examples), per_batch):
        chunk, rows, masks = examples[i:i + per_batch], [], []
        for j, e in enumerate(chunk):
            ctx, ends = enc.encode(e['ctx']), [enc.encode(' ' + end) for end in e['endings']]
            room = max_len - max(map(len, ends))
            demo = []
            if shots:
                picks = random.Random(1234 + i + j).sample(range(len(demos)), min(shots, len(demos)))
                demo = [enc.encode(demos[k]['ctx'] + ' ' + demos[k]['endings'][demos[k]['label']] + '\n\n') for k in picks]
                while demo and sum(map(len, demo)) + len(ctx) > room:     # keep inside the context window
                    demo.pop(0)
            ctx = (sum(demo, []) + ctx)[-room:]
            for et in ends:
                rows.append(ctx + et); masks.append([0] * len(ctx) + [1] * len(et))
        L = max(map(len, rows))
        tok = torch.zeros(len(rows), L, dtype=torch.long); mask = torch.zeros(len(rows), L)
        for j, (r, m) in enumerate(zip(rows, masks)):
            tok[j, :len(r)] = torch.tensor(r); mask[j, :len(m)] = torch.tensor(m)
        tok, mask = tok.to(dev), mask.to(dev)
        with autocast():
            logits = logits_fn(tok)
        losses = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), tok[:, 1:].reshape(-1), reduction='none').view(len(rows), -1)
        m = mask[:, 1:]
        s = (losses * m).sum(1)
        avg = s / m.sum(1)
        labels = torch.tensor([e['label'] for e in chunk], device=dev)
        right += (s.view(-1, 4).argmin(1) == labels).sum().item()
        ok = (avg.view(-1, 4).argmin(1) == labels).tolist()
        right_norm += sum(ok); correct_norm += ok
    return dict(acc=right / len(examples), acc_norm=right_norm / len(examples), correct_norm=correct_norm)

@torch.no_grad()
def bench_inference(model, batch_sizes=(1, 16), prompt=128, new=128):
    """Prefill and decode speed with a KV cache (eager PyTorch, no CUDA graphs).
    A 124M model is launch-bound at batch 1, so batch 16 shows far more tokens/s: Part 5 of the guide."""
    model.eval(); out = {}
    prompt, new = min(prompt, model.c.block_size // 2), min(new, model.c.block_size // 2)
    for B in batch_sizes:
        for rep in range(2):                                 # first pass warms up kernels
            caches = model.new_caches(B, prompt + new)
            x = torch.randint(0, 50257, (B, prompt), device=dev)
            sync(); t0 = time.perf_counter()
            with autocast():
                logits, _ = model(x, caches=caches, pos=0, last_only=True)
            sync(); t_pre = time.perf_counter() - t0
            nxt = logits.argmax(-1)
            t0 = time.perf_counter()
            for i in range(new):
                with autocast():
                    logits, _ = model(nxt, caches=caches, pos=prompt + i, last_only=True)
                nxt = logits.argmax(-1)
            sync(); t_dec = time.perf_counter() - t0
        f, b = model.step_cost(B, prompt + new / 2)          # an average decode step (context grows prompt → prompt+new)
        steps = new / t_dec
        out[f'batch{B}'] = dict(prefill_tok_s=B * prompt / t_pre, decode_tok_s=B * new / t_dec, ms_per_step=t_dec / new * 1e3,
                                intensity=f / b, achieved_GBps=b * steps / 1e9, achieved_TFLOPs=f * steps / 1e12,
                                bw_util=b * steps / BW if BW else None)
    model.train()
    return out

@torch.no_grad()
def generate(model, prompt, n=60, temperature=0.8, top_k=50):
    model.eval()
    idx = torch.tensor([enc.encode(prompt)[-(model.c.block_size - 1):]], device=dev)
    n = min(n, model.c.block_size - idx.size(1))             # stay inside the context window
    caches = model.new_caches(1, idx.size(1) + n)
    with autocast():
        logits, _ = model(idx, caches=caches, pos=0, last_only=True)
    out = idx[0].tolist()
    for i in range(n):
        p = logits[0, -1, :enc.n_vocab].float() / temperature   # drop the padded ids (50,257 → 50,304)
        v, ix = p.topk(top_k)
        nxt = ix[torch.multinomial(F.softmax(v, -1), 1)]
        out.append(nxt.item())
        with autocast():
            logits, _ = model(nxt.view(1, 1), caches=caches, pos=idx.size(1) + i, last_only=True)
    model.train()
    return enc.decode(out)

VAL_TOKENS = 10 * 2**20          # 10.5M tokens for the final score
VAL_TOKENS_QUICK = 2 * 2**20     # during training

def benchmark(logits_fn, val_tokens, B, model=None, T=1024):
    t0 = time.time()
    r = dict(val_loss=val_loss(logits_fn, val_tokens[:VAL_TOKENS + 1], B, T))
    r['perplexity'] = math.exp(r['val_loss'])
    ex = load_hellaswag()[:HELLASWAG_N]
    h0 = hellaswag(logits_fn, ex, max_len=T)
    r['hellaswag'], r['hellaswag_norm'], r['hellaswag_n'] = h0['acc'], h0['acc_norm'], len(ex)
    # zero-, one-, few-shot on the same first FEWSHOT_N questions, so the columns compare like with like
    n = min(len(ex), FEWSHOT_N); r['fewshot_n'] = n; r['hellaswag_shots'] = {}
    demos = load_hellaswag('train') if any(k > 0 for k in SHOTS) else None
    for k in SHOTS:
        acc = sum(h0['correct_norm'][:n]) / n if k == 0 else hellaswag(logits_fn, ex[:n], k, demos, T)['acc_norm']
        r['hellaswag_shots'][str(k)] = acc
    if model is not None:
        r['inference'] = bench_inference(model)
    r['bench_time_s'] = time.time() - t0
    return r

# %%
# Training: AdamW, warmup + cosine decay, gradient clipping, bf16 autocast, resumable checkpoints.
def pick_micro_batch(model, T, total):
    """Largest micro-batch that fits, probed in eager mode (uses more memory than compiled, so it's safe)."""
    if dev == 'cpu':
        return max(1, min(4, total // T))
    mem = torch.cuda.get_device_properties(0).total_memory
    def attempt(B):
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, model.c.vocab_size, (B, T), device=dev)
        with autocast():
            _, loss = model(x, x)
        loss.backward()
        sync()
        return torch.cuda.max_memory_allocated()
    for B in (64, 32, 16, 8, 4, 2, 1):
        if total % (B * T):
            continue
        try:
            peak = attempt(B)
        except torch.cuda.OutOfMemoryError:
            peak = float('inf')
        model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        if peak < 0.8 * mem:                                   # leave room for optimizer state
            return B
    raise RuntimeError('even micro-batch 1 does not fit')

def lr_at(step, steps, warmup):
    if step < warmup:
        return LR * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))   # cosine down to 10% of peak

def train(name, budget=None):
    budget = budget or BUDGET
    bud = BUDGETS[budget]
    tag = run_tag(name, budget)
    res_path, ck_path = f'{OUT}/results/{tag}.json', f'{OUT}/ckpt/{tag}.pt'
    if os.path.exists(res_path):
        print(f'{tag}: already done, results in {res_path}'); return json.load(open(res_path))
    train_files, val_file = get_data(bud['tokens'])
    val_tokens = torch.from_numpy(load_shard(val_file)[:VAL_TOKENS + 1].astype(np.int64))

    c = make_cfg(name); T = c.block_size
    if 'gdn' in (c.attn,):
        load_fla()
    torch.manual_seed(SEED)
    model = GPT(c).to(dev)
    B = pick_micro_batch(model, T, bud['batch'])
    accum = bud['batch'] // (B * T)
    steps = bud['tokens'] // bud['batch']
    warmup = max(30, int(0.03 * steps))
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY), dict(params=no_decay, weight_decay=0.0)],
                            lr=LR, betas=(0.9, 0.95), eps=1e-8, fused=dev == 'cuda')
    scaler = torch.amp.GradScaler(enabled=DTYPE == torch.float16)
    log = dict(name=name, budget=budget, gpu=GPU, cfg=asdict(c), params=model.n_params(),
               params_non_embedding=model.n_params(True), kv_bytes_per_token=model.kv_bytes_per_token(),
               micro_batch=B, grad_accum=accum, steps=steps, lr=LR, seed=SEED, curve=[], train_time_s=0.0)
    start, loader_state = 0, None
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        start, loader_state, log = ck['step'], ck['loader'], ck['log']
        print(f'{tag}: resuming at step {start}/{steps}')
    loader = Loader(train_files, B, T, loader_state)
    cmodel = torch.compile(model) if COMPILE and dev == 'cuda' else model
    eval_every = max(1, steps // 20); ckpt_every = max(1, steps // 10)
    print(f'{tag}: {log["params"]/1e6:.1f}M params · micro-batch {B} × {accum} accum · {steps} steps · warmup {warmup}')
    dts = []
    if dev == 'cuda': torch.cuda.reset_peak_memory_stats()
    for step in range(start, steps):
        t0 = time.perf_counter()
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, warmup)
        model.set_dense_warmup(step < int(0.05 * steps))     # DSA only: dense attention while the indexer learns
        loss_acc = 0.0
        for _ in range(accum):
            x, y = loader.next()
            with autocast():
                _, loss = cmodel(x, y)
            loss = loss / accum
            loss_acc += loss.detach()
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        sync(); dt = time.perf_counter() - t0
        log['train_time_s'] += dt
        if step - start >= 10:                                 # skip compile + warm-up steps
            dts.append(dt)
        last = step == steps - 1
        if step % eval_every == 0 or last:
            model.eval()
            vl = val_loss(lambda i: cmodel(i)[0], val_tokens[:VAL_TOKENS_QUICK + 1], B, T)
            model.train()
            log['curve'].append(dict(step=step + 1, tokens=(step + 1) * bud['batch'], train_loss=loss_acc.item(),
                                     val_loss=vl, lr=opt.param_groups[0]['lr'], time_s=log['train_time_s']))
            tps = bud['batch'] / (sorted(dts)[len(dts) // 2] if dts else dt)
            print(f'step {step+1:5d}/{steps} · train {loss_acc.item():.4f} · val {vl:.4f} · {tps/1e3:.0f}k tok/s · '
                  f'grad norm {gnorm:.2f} · {log["train_time_s"]/60:.1f} min')
        if (step + 1) % ckpt_every == 0 and not last:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1, loader=loader.state(), log=log), ck_path)
    med = sorted(dts)[len(dts) // 2] if dts else log['train_time_s'] / max(1, steps - start)
    log['tok_per_s'] = bud['batch'] / med
    log['mfu'] = log['tok_per_s'] * model.flops_per_token() / PEAK if PEAK else None
    log['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else None
    print(f'{tag}: training done · {log["tok_per_s"]/1e3:.0f}k tok/s' + (f' · MFU {log["mfu"]*100:.0f}%' if log['mfu'] else '') + ' · benchmarking…')
    # val loss uses the compiled model (fixed shapes); HellaSwag's varying lengths use the eager one (no recompiles)
    log['final'] = benchmark(lambda i: cmodel(i)[0] if i.size(1) == T else model(i)[0], val_tokens, B, model, T)
    log['sample'] = generate(model, 'The meaning of life is')
    torch.save(dict(model=model.state_dict(), cfg=asdict(c)), f'{OUT}/ckpt/{tag}_final.pt')
    if os.path.exists(ck_path):
        os.remove(ck_path)
    json.dump(log, open(res_path, 'w'), indent=1)
    f = log['final']
    shots = ' / '.join(f'{100*v:.1f}' for v in f['hellaswag_shots'].values())
    print(f'{tag}: val loss {f["val_loss"]:.4f} (ppl {f["perplexity"]:.1f}) · HellaSwag {f["hellaswag_norm"]*100:.1f}% · '
          f'{"/".join(f["hellaswag_shots"])}-shot {shots}% (on {f["fewshot_n"]} questions)')
    del cmodel, model, opt
    gc.collect()
    if dev == 'cuda': torch.cuda.empty_cache()
    return log

# %%
# Train and benchmark every run in RUNS. Re-running skips finished runs and resumes unfinished ones.
for name in RUNS:
    train(name)

# %%
# Export: package each trained model to download and use offline.
#   {OUT}/export/<run>/<run>_fp16.pt  ~250 MB, runs anywhere with torch + tiktoken (see use_offline.py)
#   {OUT}/export/<run>/hf/            Hugging Face format (gpt2, llama-style, qwen3-style, qwen3-mqa):
#                                     loads with transformers, serves with vLLM, converts to GGUF for llama.cpp
#   {OUT}/export/<run>.zip            both, in one file to download
import shutil

def load_final(tag, device=None):
    """tag = run_tag(name), e.g. 'gpt2__standard'."""
    ck = torch.load(f'{OUT}/ckpt/{tag}_final.pt', map_location='cpu', weights_only=False)
    c = Cfg(**ck['cfg'])
    if c.attn == 'gdn' and (device or dev) == 'cuda':
        load_fla()
    m = GPT(c); m.load_state_dict(ck['model'])
    return m.to(device or dev)

def hf_arch(c):
    """Which Hugging Face architecture these exact weights correspond to, or None."""
    if c.attn != 'mha' or c.hybrid_every:
        return None                                           # MLA, sliding window, DSA, GDN: no standard HF class here
    if c.pos == 'learned' and c.norm == 'layernorm' and c.mlp == 'gelu' and c.bias and not c.qk_norm and c.n_kv_head in (0, c.n_head):
        return 'gpt2'
    if c.pos == 'rope' and c.norm == 'rmsnorm' and c.mlp == 'swiglu' and not c.bias:
        return 'qwen3' if c.qk_norm else 'llama'
    return None

def to_hf(model):
    """Copy the weights into the matching transformers model (vocab trimmed to GPT-2's 50,257 real tokens)."""
    c = model.c; arch = hf_arch(c)
    if arch is None:
        return None
    import transformers
    sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    V, d, L, nh = enc.n_vocab, c.n_embd, c.n_layer, c.n_head
    nkv, hd, h = c.n_kv_head or nh, d // nh, model.blocks[0].mlp.down.in_features
    out = {}
    if arch == 'gpt2':
        cfg = transformers.GPT2Config(vocab_size=V, n_positions=c.block_size, n_embd=d, n_layer=L, n_head=nh,
                                      activation_function='gelu_new', layer_norm_epsilon=1e-5,
                                      resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, tie_word_embeddings=True)
        hf = transformers.GPT2LMHeadModel(cfg)
        out.update({'transformer.wte.weight': sd['wte.weight'][:V], 'transformer.wpe.weight': sd['wpe.weight'],
                    'transformer.ln_f.weight': sd['nf.weight'], 'transformer.ln_f.bias': sd['nf.bias']})
        for i in range(L):
            p, q = f'blocks.{i}.', f'transformer.h.{i}.'
            for a, b in (('n1', 'ln_1'), ('n2', 'ln_2')):
                out[q + b + '.weight'], out[q + b + '.bias'] = sd[p + a + '.weight'], sd[p + a + '.bias']
            # HF GPT-2 uses Conv1D, which stores weights as [in, out]: transpose
            out[q + 'attn.c_attn.weight'] = torch.cat([sd[p + 'attn.q.weight'], sd[p + 'attn.kv.weight']]).T
            out[q + 'attn.c_attn.bias'] = torch.cat([sd[p + 'attn.q.bias'], sd[p + 'attn.kv.bias']])
            out[q + 'attn.c_proj.weight'], out[q + 'attn.c_proj.bias'] = sd[p + 'attn.o.weight'].T, sd[p + 'attn.o.bias']
            out[q + 'mlp.c_fc.weight'], out[q + 'mlp.c_fc.bias'] = sd[p + 'mlp.up.weight'].T, sd[p + 'mlp.up.bias']
            out[q + 'mlp.c_proj.weight'], out[q + 'mlp.c_proj.bias'] = sd[p + 'mlp.down.weight'].T, sd[p + 'mlp.down.bias']
    else:
        common = dict(vocab_size=V, hidden_size=d, intermediate_size=h, num_hidden_layers=L, num_attention_heads=nh,
                      num_key_value_heads=nkv, head_dim=hd, max_position_embeddings=c.block_size, rms_norm_eps=1e-6,
                      rope_theta=c.rope_theta, tie_word_embeddings=True, hidden_act='silu', attention_bias=False)
        if arch == 'llama':
            hf = transformers.LlamaForCausalLM(transformers.LlamaConfig(**common, mlp_bias=False))
        else:
            hf = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**common, use_sliding_window=False))
        out.update({'model.embed_tokens.weight': sd['wte.weight'][:V], 'model.norm.weight': sd['nf.weight']})
        for i in range(L):
            p, q = f'blocks.{i}.', f'model.layers.{i}.'
            kv = sd[p + 'attn.kv.weight']                    # rows: all K heads, then all V heads
            out.update({q + 'self_attn.q_proj.weight': sd[p + 'attn.q.weight'], q + 'self_attn.k_proj.weight': kv[:nkv * hd],
                        q + 'self_attn.v_proj.weight': kv[nkv * hd:], q + 'self_attn.o_proj.weight': sd[p + 'attn.o.weight'],
                        q + 'input_layernorm.weight': sd[p + 'n1.weight'], q + 'post_attention_layernorm.weight': sd[p + 'n2.weight'],
                        q + 'mlp.gate_proj.weight': sd[p + 'mlp.up.weight'][:h], q + 'mlp.up_proj.weight': sd[p + 'mlp.up.weight'][h:],
                        q + 'mlp.down_proj.weight': sd[p + 'mlp.down.weight']})
            if arch == 'qwen3':
                out[q + 'self_attn.q_norm.weight'], out[q + 'self_attn.k_norm.weight'] = sd[p + 'attn.qn.weight'], sd[p + 'attn.kn.weight']
    missing, unexpected = hf.load_state_dict(out, strict=False)
    assert not unexpected, unexpected
    assert all(k == 'lm_head.weight' or k.endswith('attn.bias') or k.endswith('masked_bias') for k in missing), missing
    hf.tie_weights()
    hf.generation_config.eos_token_id = hf.config.eos_token_id = enc.eot_token
    # Self-check: the HF model must give the same logits as ours
    hf.eval(); model_cpu = model.float().cpu().eval()
    x = torch.randint(0, V, (2, 48))
    with torch.no_grad():
        err = (hf(x).logits - model_cpu(x)[0][..., :V]).abs().max().item()
    assert err < 1e-3, f'HF export mismatch: {err}'
    print(f'  HF {arch} export matches the trained model (max logit difference {err:.1e})')
    return hf

def export(name, budget=None):
    tag = run_tag(name, budget)
    out = f'{OUT}/export/{tag}'
    os.makedirs(out, exist_ok=True)
    model = load_final(tag, 'cpu').float().eval()
    weights = {k: v.half() for k, v in model.state_dict().items() if k != 'head.weight'}   # head is tied to wte
    torch.save(dict(variant=name, cfg=asdict(model.c), model=weights), f'{out}/{tag}_fp16.pt')
    hf = to_hf(model)
    if hf is not None:
        hf.save_pretrained(f'{out}/hf')
        from transformers import GPT2TokenizerFast
        GPT2TokenizerFast.from_pretrained('gpt2').save_pretrained(f'{out}/hf')
    zip_path = shutil.make_archive(out, 'zip', out)
    print(f'{tag}: exported to {out} ({"fp16 + HF" if hf is not None else "fp16 only: no matching HF architecture"}) · zip {os.path.getsize(zip_path)/1e6:.0f} MB')
    return zip_path

for name in RUNS:
    export(name)
# To download: open Google Drive → MyDrive/gpt2-lab/export/ → right-click the .zip → Download.
# (Or, for small files, in Colab: from google.colab import files; files.download(path).)

# %%
# Reference row: OpenAI's released GPT-2 124M (trained on ~10B+ tokens of WebText), on the same benchmark.
ref_path = f'{OUT}/results/openai-gpt2__reference.json'
if not os.path.exists(ref_path):
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained('gpt2').to(dev).eval()
    _, val_file = get_data(0)                                  # validation shard only
    vt = torch.from_numpy(load_shard(val_file)[:VAL_TOKENS + 1].astype(np.int64))
    r = benchmark(lambda i: hf(i).logits, vt, 16 if dev == 'cuda' else 1)
    json.dump(dict(name='openai-gpt2', budget='reference', gpu=GPU, params=124_439_808, final=r, curve=[]), open(ref_path, 'w'), indent=1)
    del hf; gc.collect()
print(json.load(open(ref_path))['final'])

# %%
# Compare: the table, then the curves.
import pandas as pd
import matplotlib.pyplot as plt
R = [json.load(open(p)) for p in sorted(glob.glob(f'{OUT}/results/*.json'))]
def cfg_model(r):                                             # rebuild the architecture (no weights) from a result
    with torch.device('meta'):
        return GPT(Cfg(**r['cfg']) if 'cfg' in r else Cfg())   # the OpenAI reference is plain GPT-2
rows = []
for r in R:
    f = r['final']; inf = f.get('inference', {})
    rows.append({'run': r['name'], 'budget': r['budget'], 'LR': r.get('lr'), 'params (M)': round(r['params'] / 1e6, 1),
                 'val loss': round(f['val_loss'], 4), 'ppl': round(f['perplexity'], 1),
                 'HellaSwag %': round(100 * f['hellaswag_norm'], 1),
                 **{f'HS {k}-shot %': round(100 * v, 1) for k, v in f.get('hellaswag_shots', {}).items()},
                 'train tok/s (k)': round(r['tok_per_s'] / 1e3) if r.get('tok_per_s') else None,
                 'MFU %': round(100 * r['mfu']) if r.get('mfu') else None,
                 'train min': round(r['train_time_s'] / 60, 1) if r.get('train_time_s') else None,
                 'KV KiB/token': round(r['kv_bytes_per_token'] / 1024, 1) if r.get('kv_bytes_per_token') else None,
                 'decode tok/s b1': round(inf['batch1']['decode_tok_s']) if inf else None,
                 'decode tok/s b16': round(inf['batch16']['decode_tok_s']) if inf else None,
                 **{f'AI {k}': round(v, 1) for k, v in intensity_table(cfg_model(r)).items()},
                 '% peak BW b16': round(100 * inf['batch16']['bw_util']) if inf and inf['batch16'].get('bw_util') else None})
table = pd.DataFrame(rows).sort_values('val loss')
display(table) if 'display' in globals() else print(table.to_string(index=False))
table.to_csv(f'{OUT}/results/summary.csv', index=False)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for r in R:
    if not r['curve']:
        ax[0].axhline(r['final']['val_loss'], ls='--', c='gray', label=r['name']); continue
    t = [p['tokens'] / 1e6 for p in r['curve']]; v = [p['val_loss'] for p in r['curve']]; s = [p['time_s'] / 60 for p in r['curve']]
    lbl = f"{r['name']} ({r['budget']}" + (f", lr {r['lr']:g})" if r.get('lr', DEFAULT_LR) != DEFAULT_LR else ')')
    ax[0].plot(t, v, label=lbl); ax[1].plot(s, v, label=lbl)
ax[0].set(xlabel='training tokens (M)', ylabel='val loss', title='Sample efficiency: loss vs tokens')
ax[1].set(xlabel='training minutes on ' + GPU, ylabel='val loss', title='Cost efficiency: loss vs wall-clock')
names = [r['name'] for r in R]; hs = [100 * r['final']['hellaswag_norm'] for r in R]
ax[2].barh(names, hs); ax[2].axvline(25, c='gray', ls=':', label='random (25%)')
ax[2].set(xlabel='HellaSwag acc_norm %', title='Zero-shot HellaSwag'); ax[2].set_xlim(23, max(hs + [30]) + 1)
# zoom on the part that matters: skip each run's first eval point (loss ≈ 10.8 at init)
ys = [p['val_loss'] for r in R for p in r['curve'][1:]] + [r['final']['val_loss'] for r in R]
for a in ax[:2]:
    a.set_ylim(min(ys) - 0.05, max(ys) + 0.1); a.legend(); a.grid(alpha=.3)
ax[2].legend(); plt.tight_layout(); plt.savefig(f'{OUT}/results/comparison.png', dpi=120); plt.show()

# Arithmetic intensity: each architecture's decode steps on this GPU's roofline (part 5 of the guide).
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
seen = set()
for r in R:
    key = json.dumps(r.get('cfg', {}), sort_keys=True)
    if key in seen:
        continue
    seen.add(key); m = cfg_model(r)
    xs = [m.intensity(b, 1024) for b in batches]
    line, = ax[0].plot(batches, xs, marker='o', ms=3, label=r['name'])
    if PEAK and BW:
        ax[1].loglog(xs, [min(PEAK, BW * x) / 1e12 for x in xs], marker='o', ms=3, c=line.get_color(), label=r['name'])
        for b in (1, 16):                                       # measured decode points (eager PyTorch), hollow markers
            inf = r['final'].get('inference', {}).get(f'batch{b}')
            if inf and inf.get('achieved_TFLOPs'):
                ax[1].scatter(inf['intensity'], inf['achieved_TFLOPs'], facecolors='none', edgecolors=line.get_color(), s=60)
ax[0].set(xscale='log', yscale='log', xlabel='decode batch size (1,024 tokens of context each)', ylabel='FLOPs per byte',
          title='Arithmetic intensity of a decode step')
if RIDGE:
    ax[0].axhline(RIDGE, c='gray', ls='--', label=f'{GPU} ridge point ({RIDGE:.0f})')
    x = np.logspace(-0.5, 3.5, 100)
    ax[1].loglog(x, np.minimum(PEAK, BW * x) / 1e12, c='gray', lw=2, label=f'{GPU} roofline')
    ax[1].set(xlabel='FLOPs per byte', ylabel='TFLOP/s (attainable; hollow = measured)', title=f'Roofline on {GPU}')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which='both')
else:
    ax[1].set_axis_off()
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which='both')
plt.tight_layout(); plt.savefig(f'{OUT}/results/intensity.png', dpi=120); plt.show()

# %% [markdown]
# ## Reading the results
# - **Compare at equal tokens first** (left plot). Then check **equal time** (middle): a change that lowers loss per
#   token but slows each step may not be worth it.
# - **Noise:** at this scale, two seeds of the same model differ by about 0.01–0.02 in val loss. Treat smaller gaps as ties,
#   or re-run with `SEED = 2`. HellaSwag on 10,042 questions has a standard error of about ±0.45%, so gaps under ~1.5 points
#   are not meaningful, and short runs sit close to 25–28% for every variant.
# - **Learning rate:** modern variants (RMSNorm, QK-norm) usually tolerate higher learning rates. For a fair verdict, run a
#   small sweep on the `quick` budget (e.g. `LR` in 6e-4, 1e-3, 2e-3) and compare each variant at its best LR.
# - **Zero / one / few-shot:** the `HS k-shot` columns show HellaSwag with k solved examples placed before each question
#   (same questions in every column). In the GPT-3 paper the 125M model barely improved with examples; using examples
#   in the prompt ("in-context learning") emerges with scale. A flat row is a real finding, not a bug.
# - **Arithmetic intensity** = FLOPs done per byte read from GPU memory. A batch-1 decode step does about 1 FLOP per
#   byte; the A100's ridge point is ~150-200 and the L4's ~400, so decode is memory-bound and its speed is set by bytes
#   (weights + KV cache). Batching raises intensity because the weights are read once for all sequences, until the KV
#   cache (which grows with batch) dominates the bytes. That ceiling is where GQA (3× fewer KV bytes) and MQA (12×) lift
#   intensity: compare the `AI decode b128` column and the curves in the intensity chart. Prefill is in the hundreds,
#   which is why it runs near peak FLOPs.
# - **GQA/MQA** are about inference: look at *KV KiB/token* (3× and 12× smaller) and batch-16 decode, then check how much
#   val loss they cost.
# - **OpenAI GPT-2** saw far more tokens (WebText, not FineWeb), so it's a reference point, not a same-budget competitor.

# %%
# Generate text from any finished run.
for name in RUNS:
    print(f'--- {name}\n' + generate(load_final(run_tag(name)), 'The history of the printing press') + '\n')
