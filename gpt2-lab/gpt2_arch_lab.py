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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
def autocast():                                   # mixed precision on GPU; plain fp32 on CPU (bf16 is emulated there)
    return torch.autocast(device_type=dev, dtype=DTYPE, enabled=dev == 'cuda')
def sync():
    if dev == 'cuda': torch.cuda.synchronize()
print(f'{GPU} · {DTYPE} · torch {torch.__version__}' + (f' · {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB' if dev == 'cuda' else ''))

# %%
# Config: the only cell you normally edit.
RUNS = ['gpt2', 'qwen3-style']   # any keys of VARIANTS below; the full ablation is listed there
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
# The model: GPT-2, with every modern change behind a switch.
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

class KVCache:
    def __init__(self, B, n_kv, T, hd, dtype, device):
        self.k = torch.zeros(B, n_kv, T, hd, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.k)
    def update(self, k, v, pos):
        T = k.size(2)
        self.k[:, :, pos:pos + T] = k; self.v[:, :, pos:pos + T] = v
        return self.k[:, :, :pos + T], self.v[:, :, :pos + T]

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_head, c.n_kv_head or c.n_head, c.n_embd // c.n_head
        assert c.n_embd % self.nh == 0 and self.nh % self.nkv == 0
        self.q = nn.Linear(c.n_embd, self.nh * self.hd, bias=c.bias)
        self.kv = nn.Linear(c.n_embd, 2 * self.nkv * self.hd, bias=c.bias)   # GQA shrinks this
        self.o = nn.Linear(self.nh * self.hd, c.n_embd, bias=c.bias)
        self.o.residual = True
        self.qn = RMSNorm(self.hd) if c.qk_norm else None
        self.kn = RMSNorm(self.hd) if c.qk_norm else None
    def forward(self, x, rope, cache=None, pos=0):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd)
        k, v = self.kv(x).view(B, T, 2, self.nkv, self.hd).unbind(2)
        if self.qn is not None:
            q, k = self.qn(q), self.kn(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if rope is not None:
            cos, sin = rope[0][pos:pos + T], rope[1][pos:pos + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            assert pos == 0 or T == 1, 'cache supports a prefill at pos 0, then one token at a time'
            k, v = cache.update(k, v, pos)
        if self.nkv != self.nh:                               # share each K/V head across a group of Q heads
            k = k.repeat_interleave(self.nh // self.nkv, dim=1)
            v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)   # FlashAttention on GPU
        return self.o(y.transpose(1, 2).reshape(B, T, -1))

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

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn, self.n2, self.mlp = Norm(c, c.n_embd), Attention(c), Norm(c, c.n_embd), MLP(c)
    def forward(self, x, rope, cache=None, pos=0):
        x = x + self.attn(self.n1(x), rope, cache, pos)       # pre-norm residual blocks
        return x + self.mlp(self.n2(x))

class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd) if c.pos == 'learned' else None
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.nf = Norm(c, c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.wte.weight                    # tied input/output embeddings
        if c.pos == 'rope':
            cos, sin = rope_tables(c.block_size, c.n_embd // c.n_head, c.rope_theta)
            self.register_buffer('cos', cos, persistent=False)
            self.register_buffer('sin', sin, persistent=False)
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
        rope = (self.cos, self.sin) if self.c.pos == 'rope' else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope, caches[i] if caches else None, pos)
        x = self.nf(x)
        if last_only:
            x = x[:, -1:]
        logits = self.head(x)
        loss = None if targets is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    def new_caches(self, B, T):
        c, hd = self.c, self.c.n_embd // self.c.n_head
        return [KVCache(B, c.n_kv_head or c.n_head, T, hd, DTYPE, self.wte.weight.device) for _ in range(c.n_layer)]
    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())         # tied weights counted once
        if non_embedding:
            n -= self.wte.weight.numel() + (self.wpe.weight.numel() if self.wpe is not None else 0)
        return n
    def kv_bytes_per_token(self, bytes_per_value=2):
        return 2 * self.c.n_layer * (self.c.n_kv_head or self.c.n_head) * (self.c.n_embd // self.c.n_head) * bytes_per_value
    def flops_per_token(self):                                # forward + backward, PaLM appendix B
        c = self.c
        return 6 * self.n_params() + 12 * c.n_layer * c.n_embd * c.block_size

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
# match_params: GQA/MQA remove K/V weights, so the MLP is widened to keep the total at GPT-2's size.
# Otherwise a loss gap could just mean "fewer parameters".

def count_params(c):
    with torch.device('meta'):
        return GPT(c).n_params()

def make_cfg(name):
    v = dict(VARIANTS[name]); match = v.pop('match_params', False)
    c = Cfg(**{**MODEL_BASE, **v})
    if match:
        target = count_params(Cfg(**MODEL_BASE))
        best = min(range(64, 8 * c.n_embd, 64), key=lambda h: abs(count_params(replace(c, mlp_hidden=h)) - target))
        c = replace(c, mlp_hidden=best)
    return c

for name in VARIANTS:
    c = make_cfg(name)
    with torch.device('meta'):
        m = GPT(c)
    print(f'{name:14s} {m.n_params()/1e6:6.1f}M params · KV cache {m.kv_bytes_per_token()/1024:5.1f} KiB/token · mlp_hidden {c.mlp_hidden or "auto"}')

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
        out[f'batch{B}'] = dict(prefill_tok_s=B * prompt / t_pre, decode_tok_s=B * new / t_dec, ms_per_step=t_dec / new * 1e3)
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
    m = GPT(Cfg(**ck['cfg'])); m.load_state_dict(ck['model'])
    return m.to(device or dev)

def hf_arch(c):
    """Which Hugging Face architecture these exact weights correspond to, or None."""
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
                 'decode tok/s b16': round(inf['batch16']['decode_tok_s']) if inf else None})
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
# - **GQA/MQA** are about inference: look at *KV KiB/token* (3× and 12× smaller) and batch-16 decode, then check how much
#   val loss they cost.
# - **OpenAI GPT-2** saw far more tokens (WebText, not FineWeb), so it's a reference point, not a same-budget competitor.

# %%
# Generate text from any finished run.
for name in RUNS:
    print(f'--- {name}\n' + generate(load_final(run_tag(name)), 'The history of the printing press') + '\n')
