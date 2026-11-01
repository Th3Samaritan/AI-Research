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
# Load a downloaded model; score any text.
def load_export(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
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
