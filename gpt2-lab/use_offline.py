# %% [markdown]
# # Use your trained GPT-2 offline
#
# Runs on a laptop CPU, any NVIDIA GPU, or Colab. Needs only `torch` and `tiktoken` (`transformers` for the Hugging Face route).
#
# 1. **Download:** Google Drive → `MyDrive/gpt2-lab/export/` → right-click `<run>.zip` → Download, then unzip it.
#    Inside: `<run>_fp16.pt` (~250 MB) and, for gpt2 / llama-style / qwen3-style / qwen3-mqa, an `hf/` folder.
# 2. **Command line** (this file is also a script):
#    ```
#    python use_offline.py gpt2__standard/gpt2__standard_fp16.pt --prompt "The history of the printing press"
#    python use_offline.py <run>_fp16.pt --chat            # type prompts in a loop
#    python use_offline.py <run>_fp16.pt --ppl essay.txt   # perplexity of the model on your own text
#    python use_offline.py <run>_fp16.pt --bench           # prefill / decode speed on this machine
#    ```
# 3. **Notebook:** set `MODEL_PATH` below and run the cells.
#
# This is a base model, not a chat model: it continues text rather than following instructions. A 124M model trained on
# 0.5B tokens writes fluent-looking but often wrong text; that's expected at this size.

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
# Offline: no Drive, no dataset download. HellaSwag (if you call it) is cached next to this file.
DATA = os.path.abspath('.')
OUT = DATA

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
# Load a downloaded model; score any text.
def load_export(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['cfg'].get('attn') == 'gdn':
        load_fla()                                            # GPU: fast Gated DeltaNet kernel (CPU uses the recurrence)
    m = GPT(Cfg(**ck['cfg']))
    missing, unexpected = m.load_state_dict(ck['model'], strict=False)
    assert not unexpected and set(missing) <= {'head.weight'}, (missing, unexpected)   # head is tied to wte
    print(f"{ck.get('variant', 'model')}: {m.n_params()/1e6:.1f}M params on {dev}")
    return m.float().to(dev).eval()

@torch.no_grad()
def perplexity(model, text):
    """How surprised the model is by a text: exp(mean next-token loss). Lower = more natural to the model."""
    ids = torch.tensor([enc.eot_token] + enc.encode(text))
    T, nll, n = model.c.block_size, 0.0, 0
    for s in range(0, len(ids) - 1, T):
        chunk = ids[s:s + T + 1].to(dev)
        if len(chunk) < 2:
            break
        with autocast():
            logits, _ = model(chunk[None, :-1])
        nll += F.cross_entropy(logits[0].float(), chunk[1:], reduction='sum').item()
        n += len(chunk) - 1
    return math.exp(nll / n)

# %%
# Command line: python use_offline.py MODEL.pt [--prompt ... | --chat | --ppl FILE | --bench]
if __name__ == '__main__' and 'ipykernel' not in sys.modules:
    import argparse
    ap = argparse.ArgumentParser(description='Use a GPT-2 lab model offline')
    ap.add_argument('model', help='the <run>_fp16.pt file from the export zip')
    ap.add_argument('--prompt', default='The meaning of life is')
    ap.add_argument('--tokens', type=int, default=100)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--chat', action='store_true', help='type prompts in a loop (empty line quits)')
    ap.add_argument('--ppl', help='a text file to score')
    ap.add_argument('--bench', action='store_true', help='prefill/decode speed on this machine')
    a = ap.parse_args()
    m = load_export(a.model)
    if a.ppl:
        print(f'perplexity: {perplexity(m, open(a.ppl, encoding="utf-8").read()):.2f}')
    if a.bench:
        print(json.dumps(bench_inference(m), indent=1))
    if a.chat:
        while True:
            p = input('\nprompt> ')
            if not p:
                break
            print(generate(m, p, a.tokens, a.temperature, a.top_k))
    elif not (a.ppl or a.bench):
        print(generate(m, a.prompt, a.tokens, a.temperature, a.top_k))

# %%
# In a notebook: point at the file you unzipped (in Colab you can use the Drive path directly).
MODEL_PATH = 'gpt2__standard/gpt2__standard_fp16.pt'
# MODEL_PATH = '/content/drive/MyDrive/gpt2-lab/export/gpt2__standard/gpt2__standard_fp16.pt'
if 'ipykernel' in sys.modules and os.path.exists(MODEL_PATH):
    m = load_export(MODEL_PATH)
    print(generate(m, 'The history of the printing press'))
    print('perplexity, ordinary sentence:', round(perplexity(m, 'The weather was cold, so she wore a warm coat.'), 1))
    print('perplexity, shuffled words:  ', round(perplexity(m, 'Coat warm cold she the so a was wore weather.'), 1))
    print(bench_inference(m))

# %% [markdown]
# ## The Hugging Face route
#
# For `gpt2`, `llama-style`, `qwen3-style` and `qwen3-mqa`, the export's `hf/` folder is a standard Transformers model
# (GPT2LMHeadModel, LlamaForCausalLM or Qwen3ForCausalLM), verified at export time to give the same outputs as the trained
# model. Any tool that reads Hugging Face models can use it:
#
# ```
# # serve it with vLLM (part 5 of the guide), then benchmark it with vllm bench serve
# vllm serve ./gpt2__standard/hf --max-model-len 1024
#
# # llama.cpp: convert to GGUF, quantize to 4-bit, run and score it
# python llama.cpp/convert_hf_to_gguf.py ./gpt2__standard/hf --outfile model-f16.gguf --outtype f16
# llama.cpp/build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
# llama.cpp/build/bin/llama-cli -m model-q4_k_m.gguf -p "The history of the printing press" -n 60
# llama.cpp/build/bin/llama-perplexity -m model-q4_k_m.gguf -f some_text.txt
# ```
# These commands weren't run as part of building this lab. One known wrinkle: qwen3-style's MLP width (2,432) isn't a
# multiple of 256, so llama.cpp's K-quants fall back to a different 4-bit type for those tensors (it prints a warning).

# %%
HF_DIR = 'gpt2__standard/hf'
if 'ipykernel' in sys.modules and os.path.exists(HF_DIR):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_DIR)
    hfm = AutoModelForCausalLM.from_pretrained(HF_DIR)
    ids = tok('The history of the printing press', return_tensors='pt')
    out = hfm.generate(**ids, max_new_tokens=60, do_sample=True, top_k=50, temperature=0.8)
    print(tok.decode(out[0]))
