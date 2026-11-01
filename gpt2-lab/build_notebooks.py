"""Build every notebook in this folder from gpt2_arch_lab.py, so model, training and benchmark code stay identical.

    python build_notebooks.py

writes  gpt2_arch_lab.ipynb          all-in-one: train any set of variants, compare, export
        variants/<n>_<variant>.ipynb one notebook per architecture
        compare_results.ipynb        reads every finished run: table, curves, reference GPT-2, samples
        use_offline.py / .ipynb      use a downloaded model on your own machine
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, 'gpt2_arch_lab.py'), encoding='utf-8').read()
REPO = 'https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab'

def parse(text):
    """Percent-format cells → list of (kind, source)."""
    cells = []
    for chunk in re.split(r'^# %%', text, flags=re.M)[1:]:
        head, _, body = chunk.partition('\n')
        cells.append(('markdown' if head.strip() == '[markdown]' else 'code', body.strip('\n')))
    return cells

CELLS = parse(SRC)
TAGS = {'setup': '# Setup:', 'config': '# Config:', 'data': '# Paths and data.', 'model': '# The model:',
        'variants': '# The variants.', 'bench': '# The benchmark:', 'train': '# Training:',
        'run': '# Train and benchmark every run', 'export': '# Export:', 'ref': '# Reference row:',
        'compare': '# Compare:', 'reading': '# ## Reading the results', 'generate': '# Generate text'}
C = {}
for kind, body in CELLS:
    for tag, prefix in TAGS.items():
        if body.startswith(prefix):
            C[tag] = (kind, body)
assert set(C) == set(TAGS), set(TAGS) - set(C)
INTRO = CELLS[0]

def md(text):
    return ('markdown', '\n'.join('# ' + l if l else '#' for l in text.strip('\n').split('\n')))

def code(text):
    return ('code', text.strip('\n'))

def config_with_runs(runs_line):
    kind, body = C['config']
    body = re.sub(r"^RUNS = .*$", runs_line, body, count=1, flags=re.M)
    return (kind, body)

def write_notebook(path, cells):
    nb_cells = []
    for kind, body in cells:
        lines = body.split('\n')
        if kind == 'markdown':
            lines = [l[2:] if l.startswith('# ') else l.lstrip('#') for l in lines]
        src = [l + '\n' for l in lines[:-1]] + lines[-1:]
        cell = dict(cell_type=kind, metadata={}, source=src)
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    nb = dict(nbformat=4, nbformat_minor=5, cells=nb_cells, metadata=dict(
        accelerator='GPU', colab=dict(provenance=[], gpuType='A100'),
        kernelspec=dict(name='python3', display_name='Python 3'), language_info=dict(name='python')))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(nb, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

def write_script(path, cells):
    out = []
    for kind, body in cells:
        out.append('# %% [markdown]\n' + body if kind == 'markdown' else '# %%\n' + body)
    open(path, 'w', encoding='utf-8').write('\n\n'.join(out) + '\n')

# ---------------------------------------------------------------- per-variant intros
VARIANTS = [
 ('gpt2', 'GPT-2, the 2019 baseline',
  """The reference everything else is measured against: **LayerNorm**, **learned absolute position embeddings**
(a 1,024 × 768 table), a **GELU MLP** 4× wide, **biases** on every layer, and **12-head multi-head attention**.

**Why it matters:** every other notebook changes something relative to this one, so run it first (or alongside).

**What to look at:** its val loss is the zero line for every comparison; its KV cache is 36 KiB per token
(2 × 12 layers × 12 heads × 64 dims × 2 bytes), and its decode arithmetic intensity levels off near 7 FLOPs per byte
however large the batch, because the KV cache grows with it."""),
 ('gpt2+rope', 'GPT-2 + RoPE (rotary positions)',
  """**One change:** the learned position table is removed. Instead, each query and key vector is *rotated* by an angle
that grows with its position, so the attention score between two tokens depends on how far apart they are.

**Cause → effect:** position becomes relative rather than absolute → the model can reuse the same pattern at any offset,
and there are no position parameters to learn (−0.79M). RoPE also lets a model run past its training length once its
frequencies are rescaled, which a learned table can't do.

**What to look at:** val loss vs tokens against `gpt2`, especially early in training."""),
 ('gpt2+rmsnorm', 'GPT-2 + RMSNorm',
  """**One change:** LayerNorm (subtract the mean, divide by the standard deviation, scale, shift) becomes RMSNorm
(divide by the root-mean-square, scale). No mean subtraction, no bias.

**Cause → effect:** one fewer reduction per norm and fewer parameters → slightly cheaper, and in practice just as stable.
That's why Llama, Qwen, Gemma and DeepSeek all use it.

**What to look at:** expect a near-tie in val loss; the interesting numbers are training tokens/s and whether loss
changes at all."""),
 ('gpt2+swiglu', 'GPT-2 + SwiGLU MLP',
  """**One change:** the MLP `down(gelu(up(x)))` becomes `down(silu(gate(x)) * up(x))`. To keep the same parameter count, the
hidden width drops from 4·d = 3,072 to 8/3·d = 2,048 (three matrices instead of two).

**Cause → effect:** the gate lets each hidden unit decide, per token, how much of the up-projection passes through, so the
layer computes products of features, not just thresholds. Shazeer (2020) found lower perplexity at equal parameters and
compute.

**What to look at:** val loss vs `gpt2` at equal tokens; tokens/s may dip slightly (three smaller matmuls)."""),
 ('gpt2+relu2', 'GPT-2 + squared ReLU',
  """**One change:** GELU becomes relu(x)², as in Primer (So et al., 2021) and the modded-nanogpt speedrun.

**Cause → effect:** exactly zero for negative inputs (sparse activations) and a smooth, growing curve for positive ones →
cheap, and often a little better than GELU at this scale.

**What to look at:** val loss against both `gpt2` and `gpt2+swiglu`."""),
 ('llama-style', 'Llama-style: RoPE + RMSNorm + SwiGLU, no biases',
  """**Three changes together:** RoPE positions, RMSNorm, the SwiGLU MLP, and no bias terms. This is the core of the
Llama recipe, which most open models since 2023 have followed.

**Question it answers:** do the single-change gains add up, overlap, or interfere? Compare its val-loss gain over
`gpt2` with the sum of the `gpt2+rope`, `gpt2+rmsnorm` and `gpt2+swiglu` gains.

**What to look at:** val loss vs tokens and vs wall-clock time."""),
 ('qwen3-style', 'Qwen3-style: Llama-style + QK-norm + GQA',
  """**On top of llama-style:** **QK-norm** (an RMSNorm on every head's query and key before RoPE) and **grouped-query
attention** with 12 query heads sharing 4 key/value heads. GQA removes K/V weights, so the MLP is widened (2,432) to keep
the total at GPT-2's 124M parameters, so a loss difference isn't just "fewer parameters".

**Cause → effect:** QK-norm caps how large attention logits can get → steadier training that tolerates higher learning
rates. GQA stores 3× fewer K/V vectors → a 3× smaller KV cache (12 KiB/token instead of 36), which is what lets a server
fit more users and decode faster (parts 2 and 5 of the guide).

**What to look at:** val loss vs `llama-style` (does GQA cost quality, does QK-norm win it back?), then KV KiB/token and
batch-16 decode speed, and the arithmetic-intensity columns: batched decode reaches about 20 FLOPs per byte instead
of GPT-2's 7, so the same GPU memory bandwidth produces more tokens."""),
 ('qwen3-mqa', 'Qwen3-style with multi-query attention (1 KV head)',
  """**One change from qwen3-style:** all 12 query heads share a single key/value head (MQA). The MLP is widened to 2,560
to keep 124M parameters.

**Cause → effect:** the KV cache shrinks 12× compared with GPT-2 (3 KiB/token) → the cheapest possible inference memory,
but one K/V head has to serve every query head, which usually costs some quality. That trade-off is why most recent
models settled on GQA instead.

**What to look at:** the val-loss gap to `qwen3-style` against the KV-cache saving, and the arithmetic intensity:
about 56 FLOPs per byte at batch 128, eight times GPT-2's."""),
]

COMMON_TAIL = """
**How to run:** Runtime → Change runtime type → A100 or L4 → Run all. Pick the length with `BUDGET` in the Config cell
(`smoke` ≈ minutes, `standard` ≈ 1 h on A100 / 3 h on L4). Results, checkpoints and the export go to
`MyDrive/gpt2-lab/`, the same folder every notebook in this lab uses, so after two or more runs open
[compare_results.ipynb]({repo}/compare_results.ipynb) to see them side by side.

When training ends, the model is exported to `MyDrive/gpt2-lab/export/` for download; see
[use_offline.ipynb]({repo}/use_offline.ipynb) for running it on your own machine."""

def nb_name(i, name):
    return f"{i}_{name.replace('+', '_')}.ipynb"

def variant_notebook(i, name, title, body):
    intro = md(f"# {i} · {title}\n\n{body.strip()}\n{COMMON_TAIL.format(repo=REPO)}")
    return [intro, C['setup'], config_with_runs(f"RUNS = ['{name}']            # this notebook's architecture"),
            C['data'], C['model'], C['variants'], C['bench'], C['train'], C['run'], C['export'],
            C['compare'], C['reading'], C['generate']]

# ---------------------------------------------------------------- compare notebook
COMPARE_INTRO = md(f"""# Compare all finished runs

Reads every result in `MyDrive/gpt2-lab/results/`, from any notebook and any session: the table (val loss, perplexity,
HellaSwag, speed, MFU, KV cache, decode speed), loss-vs-tokens and loss-vs-time curves, OpenAI's GPT-2 124M as a
reference row, and a text sample from every trained model. Nothing is trained here, so any GPU (or a T4) is enough.

Train variants with the [variant notebooks]({REPO}/variants/) or the [all-in-one notebook]({REPO}/gpt2_arch_lab.ipynb) first.""")
COMPARE_GEN = code("""# The same prompt through every trained model, side by side.
for p in sorted(glob.glob(f'{OUT}/ckpt/*_final.pt')):
    tag = os.path.basename(p)[:-len('_final.pt')]
    print(f'--- {tag}\\n' + generate(load_final(tag), 'The history of the printing press') + '\\n')""")
compare_cells = [COMPARE_INTRO, C['setup'], config_with_runs("RUNS = []                        # nothing is trained here"),
                 C['data'], C['model'], C['variants'], C['bench'], C['export'], C['ref'], C['compare'], C['reading'], COMPARE_GEN]

# ---------------------------------------------------------------- offline
OFF_INTRO = md("""# Use your trained GPT-2 offline

Runs on a laptop CPU, any NVIDIA GPU, or Colab. Needs only `torch` and `tiktoken` (`transformers` for the Hugging Face route).

1. **Download:** Google Drive → `MyDrive/gpt2-lab/export/` → right-click `<run>.zip` → Download, then unzip it.
   Inside: `<run>_fp16.pt` (~250 MB) and, for gpt2 / llama-style / qwen3-style / qwen3-mqa, an `hf/` folder.
2. **Command line** (this file is also a script):
   ```
   python use_offline.py gpt2__standard/gpt2__standard_fp16.pt --prompt "The history of the printing press"
   python use_offline.py <run>_fp16.pt --chat            # type prompts in a loop
   python use_offline.py <run>_fp16.pt --ppl essay.txt   # perplexity of the model on your own text
   python use_offline.py <run>_fp16.pt --bench           # prefill / decode speed on this machine
   ```
3. **Notebook:** set `MODEL_PATH` below and run the cells.

This is a base model, not a chat model: it continues text rather than following instructions. A 124M model trained on
0.5B tokens writes fluent-looking but often wrong text; that's expected at this size.""")
OFF_PATHS = code("""# Offline: no Drive, no dataset download. HellaSwag (if you call it) is cached next to this file.
DATA = os.path.abspath('.')
OUT = DATA""")
OFF_FUNCS = code("""# Load a downloaded model; score any text.
def load_export(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = GPT(Cfg(**ck['cfg']))
    missing, unexpected = m.load_state_dict(ck['model'], strict=False)
    assert not unexpected and set(missing) <= {'head.weight'}, (missing, unexpected)   # head is tied to wte
    print(f"{ck.get('variant', 'model')}: {m.n_params()/1e6:.1f}M params on {dev}")
    return m.float().to(dev).eval()

@torch.no_grad()
def perplexity(model, text):
    \"\"\"How surprised the model is by a text: exp(mean next-token loss). Lower = more natural to the model.\"\"\"
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
    return math.exp(nll / n)""")
OFF_CLI = code("""# Command line: python use_offline.py MODEL.pt [--prompt ... | --chat | --ppl FILE | --bench]
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
            p = input('\\nprompt> ')
            if not p:
                break
            print(generate(m, p, a.tokens, a.temperature, a.top_k))
    elif not (a.ppl or a.bench):
        print(generate(m, a.prompt, a.tokens, a.temperature, a.top_k))""")
OFF_NB = code("""# In a notebook: point at the file you unzipped (in Colab you can use the Drive path directly).
MODEL_PATH = 'gpt2__standard/gpt2__standard_fp16.pt'
# MODEL_PATH = '/content/drive/MyDrive/gpt2-lab/export/gpt2__standard/gpt2__standard_fp16.pt'
if 'ipykernel' in sys.modules and os.path.exists(MODEL_PATH):
    m = load_export(MODEL_PATH)
    print(generate(m, 'The history of the printing press'))
    print('perplexity, ordinary sentence:', round(perplexity(m, 'The weather was cold, so she wore a warm coat.'), 1))
    print('perplexity, shuffled words:  ', round(perplexity(m, 'Coat warm cold she the so a was wore weather.'), 1))
    print(bench_inference(m))""")
OFF_HF_MD = md("""## The Hugging Face route

For `gpt2`, `llama-style`, `qwen3-style` and `qwen3-mqa`, the export's `hf/` folder is a standard Transformers model
(GPT2LMHeadModel, LlamaForCausalLM or Qwen3ForCausalLM), verified at export time to give the same outputs as the trained
model. Any tool that reads Hugging Face models can use it:

```
# serve it with vLLM (part 5 of the guide), then benchmark it with vllm bench serve
vllm serve ./gpt2__standard/hf --max-model-len 1024

# llama.cpp: convert to GGUF, quantize to 4-bit, run and score it
python llama.cpp/convert_hf_to_gguf.py ./gpt2__standard/hf --outfile model-f16.gguf --outtype f16
llama.cpp/build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
llama.cpp/build/bin/llama-cli -m model-q4_k_m.gguf -p "The history of the printing press" -n 60
llama.cpp/build/bin/llama-perplexity -m model-q4_k_m.gguf -f some_text.txt
```
These commands weren't run as part of building this lab. One known wrinkle: qwen3-style's MLP width (2,432) isn't a
multiple of 256, so llama.cpp's K-quants fall back to a different 4-bit type for those tensors (it prints a warning).""")
OFF_HF = code("""HF_DIR = 'gpt2__standard/hf'
if 'ipykernel' in sys.modules and os.path.exists(HF_DIR):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_DIR)
    hfm = AutoModelForCausalLM.from_pretrained(HF_DIR)
    ids = tok('The history of the printing press', return_tensors='pt')
    out = hfm.generate(**ids, max_new_tokens=60, do_sample=True, top_k=50, temperature=0.8)
    print(tok.decode(out[0]))""")
offline_cells = [OFF_INTRO, C['setup'], OFF_PATHS, C['model'], C['bench'], OFF_FUNCS, OFF_CLI, OFF_NB, OFF_HF_MD, OFF_HF]

# ---------------------------------------------------------------- write everything
if __name__ == '__main__':
    write_notebook(os.path.join(HERE, 'gpt2_arch_lab.ipynb'), CELLS)
    for i, (name, title, body) in enumerate(VARIANTS, 1):
        write_notebook(os.path.join(HERE, 'variants', nb_name(i, name)), variant_notebook(i, name, title, body))
    write_notebook(os.path.join(HERE, 'compare_results.ipynb'), compare_cells)
    write_notebook(os.path.join(HERE, 'use_offline.ipynb'), offline_cells)
    write_script(os.path.join(HERE, 'use_offline.py'), offline_cells)
    print(f'built {3 + len(VARIANTS)} notebooks + use_offline.py')
