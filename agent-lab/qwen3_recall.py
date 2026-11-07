# %% [markdown]
# # Qwen3 long-context recall: can the agent's model find things in its own context?
#
# The agent lab starts where the GPT-2 lab ended. There, a recall test showed that validation loss hides how well a
# model retrieves exact details from far back, and that GQA (fewer K/V heads) cost no recall while sliding windows and
# linear attention did. Qwen3 uses the same parts as the lab's `qwen3-style` model (GQA, RoPE, QK-norm, SwiGLU,
# RMSNorm), scaled up and trained on ~36T tokens. This notebook runs the same two tests on it, at agent-sized
# contexts, and at three precisions, before any agent is built on top of it.
#
# **Why this comes first.** An agent's memory is its context window: the instructions, the tool definitions and every
# tool result so far. Each step, the model has to pull an exact name, path or number back out of that context. If it
# can't at 16K tokens, the agent fails there, whatever its other scores say.
#
# **The tests** (same design as the GPT-2 lab's `recall_test`, scaled up)
# | Test | What it does | Score |
# |---|---|---|
# | Copying | a random 32-token sequence, filler text, then the same sequence again | share of the 2nd copy predicted exactly |
# | Copy control | the same, but the first copy differs | must be ~0%: proves the score comes from memory |
# | Key lookup | 1, 4 or 16 facts like "The password for Maria is river.", filler, then "The password for Maria is" | right answer ranked first of 8 |
# | No-fact control | the question with no fact in the prompt | the guessing baseline |
#
# Each test runs at several context lengths (`CONTEXTS`) and depths (`DEPTHS`: how far back the thing to recall sits,
# 0.1 = near the end, 0.9 = near the start). Depth 0.5 checks "lost in the middle": models often recall the start and
# end of a long context better than its middle.
#
# **Precisions** (`PRECISIONS`): bf16 (as trained), 8-bit and 4-bit (bitsandbytes). Quantization shrinks the
# weights 2-4×. Perplexity usually barely moves; exact recall over long contexts is where it may quietly break.
#
# **On Colab:** Runtime → Change runtime type → **L4** → Run all. About 20-30 min per precision with the defaults.
# Results go to `MyDrive/agent-lab/recall/`; finished precisions are skipped, so Run all again resumes.

# %%
# Setup: imports, GPU detection, paths.
import os, sys, json, time, glob, gc, random, subprocess
import numpy as np

IN_COLAB = 'google.colab' in sys.modules
def pip(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=True)
if IN_COLAB:
    pip('tiktoken', 'bitsandbytes', 'accelerate')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU'
print(f'{GPU} · torch {torch.__version__}')
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT = '/content/drive/MyDrive/agent-lab'
else:
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.', 'out')
os.makedirs(f'{OUT}/recall', exist_ok=True)

# %%
# Config: the only cell you normally edit.
MODEL = os.environ.get('MODEL', 'Qwen/Qwen3-1.7B')   # or a local folder; Qwen3-0.6B / 4B / 8B work the same way
PRECISIONS = ['bf16', '8bit', '4bit']         # 8bit and 4bit need a CUDA GPU (bitsandbytes)
CONTEXTS = [1024, 4096, 8192, 16384]          # 32768 also fits on an L4 at 1.7B, but takes ~4x longer than 16384
DEPTHS = (0.1, 0.5, 0.9)                      # how far back the thing to recall sits, as a share of the context
KV_FACTS = (1, 4, 16)                         # facts in the prompt: memory load
TRIALS = 30                                   # per condition; ±9 points at 50% accuracy, enough to see a collapse
COPY_LEN = 32                                 # tokens in the sequence to copy
CHOICES = 8                                   # lookup answers to pick from: chance = 12.5%
SEED = 2026
FILLER_FILE = os.environ.get('FILLER_FILE')   # a .txt file to use as filler instead of FineWeb (offline tests)
if not IN_COLAB and dev == 'cpu':             # a quick local check on a laptop CPU
    PRECISIONS, CONTEXTS, TRIALS = ['bf16'], [256, 512], 2

# %%
# Filler text: held-out FineWeb, the same text the GPT-2 lab validated on. It arrives as GPT-2 tokens, so it is decoded
# back to text and re-tokenized with Qwen's tokenizer.
tok = AutoTokenizer.from_pretrained(MODEL)

def load_filler(n_gpt2_tokens=2_000_000):
    if FILLER_FILE:
        text = open(FILLER_FILE, encoding='utf-8').read()
    else:
        import tiktoken
        from huggingface_hub import hf_hub_download
        p = hf_hub_download('kjj0/fineweb10B-gpt2', 'fineweb_val_000000.bin', repo_type='dataset')
        with open(p, 'rb') as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            assert header[0] == 20240520, 'unexpected shard format'
            ids = np.frombuffer(f.read(2 * min(int(header[2]), n_gpt2_tokens)), dtype=np.uint16)
        gpt2 = tiktoken.get_encoding('gpt2')
        docs = np.split(ids, np.where(ids == gpt2.eot_token)[0])
        text = '\n\n'.join(gpt2.decode(d[d != gpt2.eot_token].tolist()) for d in docs)
    return tok(text, add_special_tokens=False)['input_ids']

FILLER = load_filler()
print(f'{len(FILLER):,} filler tokens')

# %%
# Build the test cases. Every precision answers exactly the same questions (fixed seed).
def single_token(words):
    return [w for w in words if len(tok.encode(' ' + w, add_special_tokens=False)) == 1]

NAMES = single_token('''Maria James Sarah David Anna John Emma Peter Laura Michael Alice Robert Julia Thomas Grace Daniel
Helen Mark Linda Paul Karen Steven Susan Kevin Nancy Brian Lisa Jason Amy Eric Kate Adam Rachel Ryan Jessica Andrew
Emily Scott Claire Henry Diana Oliver Sophie Martin Rose Victor Nina Simon'''.split())
VALUES = single_token('''river apple stone cloud tiger garden silver window candle forest orange planet rocket bridge
castle dragon engine falcon guitar harbor island jacket ladder marble needle ocean pencil rabbit saddle tunnel violin
wagon yellow anchor basket camera desert feather hammer kettle lemon mirror nickel parrot pepper shadow summer thunder
velvet winter copper cotton dolphin'''.split())

def build_cases(ctx):
    rng = random.Random(f'{SEED}-{ctx}')
    counts = np.bincount(np.array(FILLER), minlength=len(tok))
    common = [t for t in np.argsort(-counts)[:20000].tolist() if counts[t] > 0 and t < tok.vocab_size]
    def filler(n):
        s = rng.randrange(len(FILLER) - n - 1)
        return FILLER[s:s + n]
    cases = {}
    for depth in DEPTHS:                                      # copying: test and control
        gap = int(depth * (ctx - 2 * COPY_LEN))               # tokens between the two copies
        test, ctrl = [], []
        for _ in range(TRIALS):
            span, other = rng.sample(common, COPY_LEN), rng.sample(common, COPY_LEN)
            pre, mid = filler(ctx - 2 * COPY_LEN - gap), filler(gap)
            test.append(pre + span + mid + span); ctrl.append(pre + other + mid + span)
        cases[f'copy|{depth}'], cases[f'copyctl|{depth}'] = test, ctrl
    def enc(s):
        return tok.encode(s, add_special_tokens=False)
    def kv_case(n_facts, depth):
        names, vals = rng.sample(NAMES, max(n_facts, 1)), rng.sample(VALUES, CHOICES + n_facts)
        target = rng.randrange(max(n_facts, 1))
        facts = sum((enc(f' The password for {names[i]} is {vals[i]}.') for i in range(n_facts)), [])
        query = enc(f'\n\nQuestion: what is the password for {names[target]}? The password for {names[target]} is')
        wrong = [vals[i] for i in range(n_facts) if i != target][:CHOICES - 1]   # other answers in the prompt first
        wrong += vals[n_facts:n_facts + CHOICES - 1 - len(wrong)]
        choices = [enc(' ' + v)[0] for v in [vals[target]] + wrong]
        gap = int(depth * (ctx - len(facts) - len(query)))    # filler between the facts and the question
        return filler(ctx - len(facts) - len(query) - gap) + facts + filler(gap) + query, choices
    for n in KV_FACTS:
        for depth in DEPTHS:
            cases[f'kv{n}|{depth}'] = [kv_case(n, depth) for _ in range(TRIALS)]
    cases['kvctl'] = [kv_case(0, 0.5) for _ in range(TRIALS)]
    return cases

print(f'{len(NAMES)} names, {len(VALUES)} answer words')

# %%
# Run the tests on one loaded model. Only the logits the score needs are computed (logits_to_keep): at 16K tokens the
# full logit matrix would be 16,384 × 151,936 values, about 5 GB.
@torch.no_grad()
def score(model, cases):
    r = {}
    for key, items in cases.items():
        right = 0.0
        for item in items:
            if key.startswith('copy'):
                x = torch.tensor([item], device=model.device)
                lg = model(x, logits_to_keep=COPY_LEN).logits[0, 1:-1]        # predictions for the 2nd copy's tokens 3..32
                right += (lg.argmax(-1) == x[0, -COPY_LEN + 2:]).float().mean().item()
            else:
                seq, choices = item
                lg = model(torch.tensor([seq], device=model.device), logits_to_keep=1).logits[0, -1]
                right += float(lg[choices].argmax().item() == 0)
        r[key] = right / len(items)
    return r

import logging
logging.getLogger('bitsandbytes').setLevel(logging.ERROR)   # its cast warning fires on every matmul and floods the output

def load(precision):
    # 8-bit matmuls run in fp16, so load 8-bit in fp16: no bf16 -> fp16 cast on every layer
    kw = dict(dtype=torch.float16 if precision == '8bit' else torch.bfloat16, device_map=dev if dev == 'cuda' else None)
    if precision != 'bf16':
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = (BitsAndBytesConfig(load_in_8bit=True) if precision == '8bit' else
                                     BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                        bnb_4bit_compute_dtype=torch.bfloat16))
    return AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()

# %%
# Test every precision. Finished ones are skipped, so Run all again resumes.
name = MODEL.rstrip('/').split('/')[-1]
for precision in PRECISIONS:
    path = f'{OUT}/recall/{name}__{precision}.json'
    if os.path.exists(path):
        print(f'{name} {precision}: already tested'); continue
    model, res, t0 = load(precision), {}, time.time()
    mem = torch.cuda.max_memory_allocated() / 2**30 if dev == 'cuda' else float('nan')
    for ctx in CONTEXTS:
        t1 = time.time()
        res[str(ctx)] = score(model, build_cases(ctx))
        c = res[str(ctx)]
        print(f'{name} {precision} @{ctx:>6}: copy far {100*c[f"copy|{DEPTHS[-1]}"]:.0f}% · '
              f'16 facts far {100*c.get(f"kv{KV_FACTS[-1]}|{DEPTHS[-1]}", float("nan")):.0f}% · {time.time()-t1:.0f} s')
    json.dump(dict(model=name, precision=precision, depths=DEPTHS, trials=TRIALS, gpu=GPU, results=res,
                   weights_gib=mem, minutes=(time.time() - t0) / 60), open(path, 'w'), indent=1)
    del model; gc.collect()
    if dev == 'cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

# %%
# Results: the table, then the charts.
import pandas as pd
import matplotlib.pyplot as plt
R = [json.load(open(p)) for p in sorted(glob.glob(f'{OUT}/recall/*.json'))]
rows = []
for r in R:
    for ctx, c in r['results'].items():
        row = dict(model=r['model'], precision=r['precision'], context=int(ctx))
        for d in r['depths']:
            row[f'copy @{d}'] = round(100 * c[f'copy|{d}'], 1)
        row['copy control'] = round(100 * np.mean([c[f'copyctl|{d}'] for d in r['depths']]), 1)
        for k, v in c.items():
            if k.startswith('kv') and k != 'kvctl':
                n, d = k[2:].split('|'); row[f'{n} facts @{d}'] = round(100 * v, 1)
        row['no fact'] = round(100 * c['kvctl'], 1)
        rows.append(row)
table = pd.DataFrame(rows).sort_values(['model', 'precision', 'context'])
display(table) if 'display' in globals() else print(table.to_string(index=False))
table.to_csv(f'{OUT}/recall/summary.csv', index=False)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
styles = {'bf16': '-', '8bit': '--', '4bit': ':'}
for r in R:
    ctxs = sorted(int(c) for c in r['results'])
    for i, d in enumerate(r['depths']):
        ax[0].plot(ctxs, [100 * r['results'][str(c)][f'copy|{d}'] for c in ctxs], styles.get(r['precision'], '-'),
                   color=f'C{i}', marker='o', label=f"{r['precision']} · depth {d}")
        n = max(int(k[2:].split('|')[0]) for k in r['results'][str(ctxs[0])] if k.startswith('kv') and k != 'kvctl')
        ax[1].plot(ctxs, [100 * r['results'][str(c)][f'kv{n}|{d}'] for c in ctxs], styles.get(r['precision'], '-'),
                   color=f'C{i}', marker='o', label=f"{r['precision']} · depth {d}")
for a, t in zip(ax, ['Copying a 32-token sequence', f'Key lookup with {KV_FACTS[-1]} facts (chance {100/CHOICES:.1f}%)']):
    a.set(xscale='log', xlabel='context length (tokens)', ylabel='% correct', ylim=(0, 102), title=t)
    a.set_xticks(CONTEXTS, [f'{c//1024}K' for c in CONTEXTS]); a.grid(alpha=.3); a.legend(fontsize=7, ncol=3)
plt.tight_layout(); plt.savefig(f'{OUT}/recall/recall.png', dpi=120); plt.show()

# %% [markdown]
# ## Reading the results
# - **Controls first.** The copy control must be near 0%, and "no fact" near 12.5% (a model may favour some words, so
#   15-20% is normal). If not, the test is broken, not the model.
# - **Context length.** A flat line up to 16K means retrieval holds at agent-sized contexts. A drop at one length is
#   where an agent built on this model would start losing track of its own earlier steps.
# - **Depth.** Lower scores at depth 0.5 than at 0.1 and 0.9 = "lost in the middle". An agent can work around it by
#   keeping key facts (the task, the plan) at the start or repeating them near the end.
# - **16 facts vs 1 fact.** The memory-load test: an agent's context holds many similar items (file names, IDs).
# - **Precision.** The gap between bf16 and 4-bit, at the longest context and 16 facts, is what quantization costs this
#   model for agent work. A gap of more than ~15 points is real at 30 trials; smaller ones are noise.
