# %% [markdown]
# # Fine-tuning the agent: QLoRA on tool-use traces
#
# The untuned agent (`mini_agent`) called its tools correctly, then restated the whole table instead of answering, and
# ran out of tokens. Its recall is not the problem: the Qwen3 recall test scored 100% at that context length. The
# problem is behaviour, and behaviour is what fine-tuning changes. This notebook teaches Qwen3-1.7B one habit: find the
# file, read it, then answer in one sentence with the number and the file it came from.
#
# | Step | What happens | Why |
# |---|---|---|
# | Baseline | run the untuned model on held-out tasks and score it | without a number from before, there is no way to tell whether training helped |
# | Data | generate synthetic labs: random results tables in random folders, questions computed from the tables, and the ideal trace for each (list → read → answer) | the right answer is known exactly; random names and numbers teach the skill, not the answers |
# | Masking | train only on the model's own turns, not on the question or the tool results | those are given to the model, not written by it; training on them would teach it to invent tool output |
# | QLoRA | base weights frozen in 4-bit; small trainable LoRA matrices (rank 16) added to every attention and MLP projection | about 1% of the weights train, so it fits on one T4 or L4 |
# | Eval | the same tasks with the trained adapter, plus questions about your real lab files | the real files were never in training, so they test whether the habit transfers |
#
# **Version 2.** The first run (v1) doubled accuracy on synthetic tasks (35% → 70%) but fell from 17% to 0% on the real
# files: every synthetic folder held exactly one table, so the model learned "list, read the only file, answer" and,
# faced with a real folder of per-run JSON files, read the wrong one or none and put file sizes in its answers. It also
# picked wrong maxima, comparing a dozen numbers by eye. v2 changes the data and the tools, not the training:
# - **Decoys.** Each table folder also holds per-run JSON files full of numbers, plots and sometimes a second CSV, as in
#   the real lab. The model has to pick the table from the listing.
# - **Recovery.** Some traces open a JSON file first, find it lacks the column, then open the table. The wrong step is
#   context only, never a training target: the model learns to recover, not to make the mistake.
# - **A sort tool.** `sort_table(path, column)` returns the column in order, so a highest/lowest question becomes
#   copying the first row. Agents hand arithmetic to tools; the model only has to choose the call.
# - **Shorter training.** v1's loss flattened by step 40 of 220; v2 trains about 100 larger steps.
# The real-file questions are the same as v1's, so those results compare directly. The synthetic eval tasks are new
# (v1 drew them in an order that changed between runs, now fixed) and their labs contain decoys.
#
# **On Colab:** Runtime → Change runtime type → **L4** → Run all, about 1 hour in total. A T4 also works, in fp16, more
# slowly. Results go to `MyDrive/agent-lab/finetune_v2/`; finished steps are skipped when you run it again.

# %%
# Config.
import os, sys
IN_COLAB = 'google.colab' in sys.modules
MODEL = os.environ.get('MODEL', 'Qwen/Qwen3-1.7B')
LAB = os.environ.get('AGENT_ROOT', '/content/drive/MyDrive/gpt2-lab' if IN_COLAB else '.')     # real files, for the final eval
OUT = os.environ.get('AGENT_OUT', '/content/drive/MyDrive/agent-lab' if IN_COLAB else 'out')
WORK = '/content/synth' if IN_COLAB else os.path.abspath('synth')                            # synthetic labs, rebuilt each run
VERSION = 'v2'           # results go to OUT/finetune_<VERSION>
N_TRAIN = 400            # training tasks; each gives one example per model turn (about 3.5)
N_EVAL = 20              # held-out synthetic tasks; the real-file tasks come on top
LORA_R, LORA_ALPHA = 16, 32
LR, EPOCHS, ACCUM = 2e-4, 1, 16  # ACCUM examples per optimizer step
MAX_LEN = 3072           # longer examples are skipped
MAX_STEPS = 6            # agent replies allowed per eval task
MAX_NEW_TOKENS = 384     # per reply in the eval: a right answer is one or two sentences
SEED = 2026

# %%
# Setup.
import subprocess, csv, json, math, random, re, time
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'bitsandbytes', 'peft'], check=True)
    from google.colab import drive
    drive.mount('/content/drive')
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
SMOKE = dev == 'cpu'     # no GPU: a tiny random model and a few tasks, to check the code end to end
if SMOKE:
    N_TRAIN, N_EVAL, MAX_NEW_TOKENS, ACCUM = 3, 2, 12, 2
BF16 = dev == 'cuda' and torch.cuda.is_bf16_supported()
compute = torch.bfloat16 if BF16 or SMOKE else torch.float16
FT = f'{OUT}/finetune_{VERSION}' + ('_smoke' if SMOKE else '')
os.makedirs(FT, exist_ok=True)
print(f'{dev} · {torch.cuda.get_device_name() if dev == "cuda" else "CPU smoke test"} · compute {compute}')

# %%
from agent_core import *   # build_notebooks.py pastes agent_core.py here

# %% [markdown]
# ## Synthetic labs
# Each lab is a small folder like your Drive one: two or three results tables in subfolders, plus a README, notes and
# run files to ignore. Run names, metrics and numbers are random. Folder names never include `results` or `recall`, so
# the real-file eval is a fair test of transfer.

# %%
NAME_A = 'alpha beta gamma delta sigma omega tiny small wide deep fast lite base mini dense sparse long flat swift core'.split()
NAME_B = 'net former mix glu norm rope attn conv moe mqa gqa hybrid relu gelu v2 v3 xl kv'.split()
METRICS = [('val loss', 2.5, 4.5, 4), ('train loss', 2.0, 4.0, 4), ('perplexity', 10, 90, 1), ('accuracy %', 20, 95, 1),
           ('F1 %', 20, 95, 1), ('BLEU', 5, 45, 1), ('exact match %', 5, 80, 1), ('recall @1K %', 0, 100, 1),
           ('copy score %', 0, 100, 1), ('tok/s (k)', 20, 250, 0), ('latency ms', 3, 250, 1), ('memory GiB', 0.5, 40, 2),
           ('params (M)', 10, 900, 1), ('MFU %', 5, 60, 0), ('train min', 5, 300, 1), ('KV KiB/token', 1, 64, 1),
           ('score', 0, 1, 3), ('win rate %', 0, 100, 1), ('steps', 500, 20000, 0), ('grad norm', 0.1, 5, 3)]
CATEGORIES = [('family', ['dense', 'sparse', 'hybrid', 'baseline']), ('budget', ['quick', 'standard', 'long']),
              ('optimizer', ['adamw', 'lion', 'sgd'])]
TABLES = [('eval', 'scores.csv', 'evaluation scores'), ('benchmarks', 'speed.csv', 'speed benchmarks'),
          ('ablations', 'table.csv', 'ablation runs'), ('sweeps', 'lr_sweep.csv', 'learning-rate sweep'),
          ('baselines', 'baselines.csv', 'baseline runs'), ('probes', 'probe_results.csv', 'probing results'),
          ('finetune', 'metrics.csv', 'fine-tuning metrics'), ('long_context', 'summary.csv', 'long-context tests')]

def run_names(rng, n):
    names = set()
    while len(names) < n:
        a, b, k = rng.choice(NAME_A), rng.choice(NAME_B), rng.randint(1, 64)
        names.add(rng.choice([f'{a}-{b}', f'{a}+{b}', f'{b}-{k}', f'{a}{k}', f'{a}-{b}-{k}']))
    return sorted(sorted(names), key=lambda _: rng.random())   # sort first: a set's order changes between runs

def fmt(v, d):
    return f'{v:.{d}f}' if d else str(int(round(v)))

def write_decoys(drng, folder, runs, table_file):
    """Files that sit next to a table, as in the real lab: per-run JSON logs full of numbers, a plot, sometimes a second
    CSV of training curves. Returns the JSON names. Uses its own random generator, so the decoys do not change which
    questions get asked."""
    names = []
    for r in drng.sample(runs, drng.randint(4, 14)):
        name = drng.choice([f'{r}.json', f'{r}__standard.json', f'{r}__seed{drng.randint(1, 3)}.json'])
        log = dict(name=r, budget=drng.choice(['quick', 'standard']), steps=drng.randint(500, 5000),
                   final=dict(val_loss=round(drng.uniform(2.5, 4.5), 4), tok_per_s=drng.randint(40000, 250000)),
                   curve=[round(drng.uniform(2.5, 9), 3) for _ in range(drng.randint(10, 40))])
        with open(os.path.join(folder, name), 'w') as f:
            json.dump(log, f)
        names.append(name)
    with open(os.path.join(folder, os.path.splitext(table_file)[0] + '.png'), 'wb') as f:
        f.write(drng.randbytes(drng.randint(20000, 90000)))
    if drng.random() < 0.3:
        with open(os.path.join(folder, 'curves.csv'), 'w') as f:
            f.write('step,loss,lr\n' + ''.join(f'{s},{drng.uniform(2.5, 9):.3f},{1e-3:.0e}\n' for s in range(0, 2000, 100)))
    return sorted(set(names))

def make_lab(rng, root):
    """Write one synthetic lab to `root` and return its tables (folder, file, topic, rows)."""
    os.makedirs(root, exist_ok=True)
    runs = run_names(rng, 18)
    tables = []
    for folder, file, topic in rng.sample(TABLES, rng.randint(2, 3)):
        while True:
            rows_runs = rng.sample(runs, rng.randint(5, 16))
            metrics = rng.sample(METRICS, rng.randint(3, 10))
            cats = rng.sample(CATEGORIES, rng.randint(0, 1))
            header = ['run'] + [c for c, _ in cats] + [m[0] for m in metrics]
            rows = []
            for r in rows_runs:
                row = {'run': r, **{c: rng.choice(v) for c, v in cats}}
                for name, lo, hi, d in metrics:
                    row[name] = '' if rng.random() < 0.04 else fmt(rng.uniform(lo, hi), d)
                rows.append(row)
            buf = [','.join(header)] + [','.join(row[h] for h in header) for row in rows]
            if len('\n'.join(buf)) < READ_LIMIT - 200:
                break
        os.makedirs(os.path.join(root, folder), exist_ok=True)
        with open(os.path.join(root, folder, file), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, header)
            w.writeheader()
            w.writerows(rows)
        decoys = write_decoys(random.Random(f'{root}/{folder}'), os.path.join(root, folder), runs, file)
        tables.append(dict(root=root, folder=folder, file=file, topic=topic, rows=rows, decoys=decoys,
                           numeric=[m[0] for m in metrics], decimals={m[0]: m[3] for m in metrics}))
    with open(os.path.join(root, 'README.md'), 'w') as f:
        f.write('# Lab\n\nTraining and evaluation runs. Tables are in the subfolders; run logs are in runs/.\n')
    with open(os.path.join(root, 'notes.txt'), 'w') as f:
        f.write(f'TODO: rerun {rng.choice(runs)} with a second seed.\nCheck the {rng.choice(METRICS)[0]} numbers.\n')
    os.makedirs(os.path.join(root, 'runs'), exist_ok=True)
    for r in rng.sample(runs, 3):
        with open(os.path.join(root, 'runs', f'{r}.json'), 'w') as f:
            json.dump(dict(name=r, seed=rng.randint(0, 9999), lr=rng.choice([3e-4, 6e-4, 1e-3])), f)
    return tables

# %% [markdown]
# ## Tasks and their ideal traces
# A task is a question, the exact strings a right answer must contain (`gold`), and the ideal trace: the replies a good
# agent would write, in order. A reply is either one tool call or the final answer, and carries a flag: trained on, or
# context only (the wrong first step of a recovery trace).

# %%
def num(s):
    return float(s)

def decimals(table, col):
    if col in table.get('decimals', {}):
        return table['decimals'][col]
    vals = [r[col] for r in table['rows'] if r[col]]
    return max((len(v.split('.')[1]) if '.' in v else 0) for v in vals)

def tool_call(name, train=True, **args):
    """A reply that calls one tool, written exactly as Qwen3 writes it."""
    return ('<tool_call>\n' + json.dumps({'name': name, 'arguments': args}, ensure_ascii=False) + '\n</tool_call>',
            [dict(name=name, arguments=args)], train)

def make_task(rng, table, kinds):
    """One question about `table`, with gold strings and the ideal trace. Returns None if the table can't support it."""
    rows = table['rows']
    path = f"{table['folder']}/{table['file']}"
    cols = [c for c in table['numeric'] if c in rows[0] and sum(bool(r[c]) for r in rows) >= 4]
    if not cols:
        return None
    col = rng.choice(cols)
    have = [r for r in rows if r[col]]
    kind = rng.choices(*zip(*kinds.items()))[0]
    explicit = rng.random() < 0.3
    where = path if explicit else f'the {table["topic"]}'
    d = decimals(table, col)
    if kind == 'max':
        word = rng.choice(['highest', 'lowest'])
        ranked = sorted(have, key=lambda r: num(r[col]), reverse=word == 'highest')
        if num(ranked[0][col]) == num(ranked[1][col]):
            return None
        best = ranked[0]
        q = rng.choice([f'Which run has the {word} {col} in {where}?', f'In {where}, which run has the {word} {col}?',
                        f'Find the run with the {word} {col} in {where}.'])
        gold = [best['run'], best[col]]
        answer = rng.choice([f'{best["run"]} has the {word} {col}: {best[col]} ({path}).',
                             f'The {word} {col} is {best[col]}, from {best["run"]} ({path}).'])
    elif kind == 'lookup':
        r = rng.choice(have)
        q = rng.choice([f"What is {r['run']}'s {col} in {where}?", f'In {where}, what {col} did {r["run"]} get?'])
        gold = [r[col]]
        answer = f"{r['run']} has a {col} of {r[col]} ({path})."
    elif kind == 'compare':
        a, b = rng.sample(have, 2)
        if num(a[col]) == num(b[col]):
            return None
        hi = a if num(a[col]) > num(b[col]) else b
        diff = fmt(abs(num(a[col]) - num(b[col])), d)
        q = rng.choice([f'How does {a["run"]} compare with {b["run"]} on {col} in {where}?',
                        f'In {where}, is {a["run"]} or {b["run"]} higher on {col}?'])
        gold = [a[col], b[col]]
        answer = f'{a["run"]} has a {col} of {a[col]} and {b["run"]} has {b[col]}, so {hi["run"]} is higher by {diff} ({path}).'
    else:   # 'missing': the run is not in the table; the right answer says so instead of inventing a number
        fake = next(n for n in run_names(rng, 40) if n not in {r['run'] for r in rows})
        q = f"What is {fake}'s {col} in {where}?"
        gold = ['no run']
        answer = f'There is no run named {fake} in {path}. It lists: {", ".join(r["run"] for r in rows)}.'
    trace = []
    trng = random.Random(q)      # trace choices get their own generator, so they don't change later questions
    if explicit and rng.random() < 0.25 and kind != 'missing':   # a wrong path in the question: recover by listing
        stem, ext = os.path.splitext(table['file'])
        wrong = f"{table['folder']}/{stem.rstrip('s') if stem.endswith('s') else stem + 's'}{ext}"
        q = q.replace(path, wrong)
        trace += [tool_call('read_file', path=wrong), tool_call('list_files', folder=table['folder'])]
    elif not explicit:
        trace += [tool_call('list_files', folder='.'), tool_call('list_files', folder=table['folder'])]
        if table.get('decoys') and trng.random() < 0.3:   # opens a run log first: context only, never a target
            trace.append(tool_call('read_file', train=False, path=f"{table['folder']}/{trng.choice(table['decoys'])}"))
    if kind == 'max':
        trace.append(tool_call('sort_table', path=path, column=col, descending=word == 'highest'))
    else:
        trace.append(tool_call('read_file', path=path))
    trace.append((answer, None, True))
    return dict(root=table['root'], path=path, kind=kind, question=q, gold=gold, trace=trace)

def synthetic_tasks(n, seed, kinds, name):
    rng, tasks = random.Random(seed), []
    while len(tasks) < n:
        lab = make_lab(rng, f'{WORK}/{name}/lab{len(tasks):04d}')
        t = make_task(rng, rng.choice(lab), kinds)
        if t:
            tasks.append(dict(t, set='synthetic'))
    return tasks

TRAIN_KINDS = {'max': 0.35, 'lookup': 0.25, 'compare': 0.3, 'missing': 0.1}
EVAL_KINDS = {'max': 0.4, 'lookup': 0.3, 'compare': 0.3}
train_tasks = synthetic_tasks(N_TRAIN, SEED, TRAIN_KINDS, 'train')
eval_tasks = synthetic_tasks(N_EVAL, SEED + 1, EVAL_KINDS, 'eval')
t = train_tasks[0]
print(f'{len(train_tasks)} training tasks, {len(eval_tasks)} synthetic eval tasks. Example:\n{t["question"]}')
for reply, _, train in t['trace']:
    print('  ' + reply.replace('\n', ' ') + ('' if train else '   (context only)'))

# %%
# Real-file tasks: the same question types about your lab's own tables, with gold answers read from the files.
REAL_TABLES = [('results', 'summary.csv', 'training results', ['val loss', 'HellaSwag %', 'train tok/s (k)', 'KV KiB/token', 'train min']),
               ('recall', 'summary.csv', 'recall test results', ['copy near %', 'copy far %', '1 fact @32', '4 facts @32', '16 facts @32'])]
rng = random.Random(SEED + 2)
real_tasks = []
for folder, file, topic, cols in REAL_TABLES:
    p = os.path.join(LAB, folder, file)
    if not os.path.exists(p):
        print(f'skipped {folder}/{file}: not found in {LAB}')
        continue
    rows = list(csv.DictReader(open(p, encoding='utf-8')))
    table = dict(root=LAB, folder=folder, file=file, topic=topic, rows=rows, numeric=[c for c in cols if c in rows[0]])
    for _ in range(200):
        if sum(t['path'] == f'{folder}/{file}' for t in real_tasks) == 6:
            break
        t = make_task(rng, table, EVAL_KINDS)
        if t and t['question'] not in {x['question'] for x in real_tasks}:
            real_tasks.append(dict(t, set='real'))
print(f'{len(real_tasks)} real-file tasks')
for t in real_tasks[:3]:
    print(' ', t['question'], '→', t['gold'])

# %% [markdown]
# ## The model: 4-bit base
# NF4 stores each weight in 4 bits, so the 1.7B model takes about 1.2 GB. Matmuls still run in bf16 (fp16 on a T4):
# weights are unpacked on the fly. The recall test showed 4-bit loses nothing measurable for this model.

# %%
tok = AutoTokenizer.from_pretrained(MODEL)
if SMOKE:
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.update(dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16))
    model = AutoModelForCausalLM.from_config(cfg).to(compute)
else:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, dtype=compute, device_map={'': 0})
model.eval()
print(f'{MODEL}: {torch.cuda.memory_allocated() / 2**30:.2f} GiB on GPU' if dev == 'cuda' else 'tiny random model')

def render(messages):
    return tok.apply_chat_template(messages, tools=list(TOOLS.values()), add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)

@torch.no_grad()
def generate(messages):
    ids = tok(render(messages), return_tensors='pt', add_special_tokens=False).to(dev)
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True), ids['input_ids'].shape[1]

# %% [markdown]
# ## Scoring
# A task is correct when the final answer contains every gold string: the run name and the number, as written in the
# file. Matching is exact, so `gpt2` does not count as a mention of `gpt2+rope`, and `3.741` not as `13.7415`.

# %%
def mentions(text, s):
    return re.search(r'(?<![\w.+\-])' + re.escape(s) + r'(?![\w+\-]|\.\d)', text, re.I) is not None

def files_read(messages):
    return [c['function']['arguments'].get('path', '').lstrip('./') for m in messages if m['role'] == 'assistant'
            for c in m.get('tool_calls', []) if c['function']['name'] in ('read_file', 'sort_table')]

def evaluate(tasks, label):
    path = f'{FT}/eval_{label}.json'
    if os.path.exists(path):
        print(f'{label}: loaded from {path}')
        return json.load(open(path))
    rows, t0 = [], time.time()
    for i, t in enumerate(tasks):
        set_root(t['root'])
        r = run_agent(t['question'], generate, max_steps=MAX_STEPS, verbose=False)
        ans = r['answer'] or ''
        rows.append(dict(set=t['set'], kind=t['kind'], question=t['question'], gold=t['gold'], answer=ans,
                         correct=all(mentions(ans, g) for g in t['gold']), finished=r['answer'] is not None,
                         found=t['path'] in files_read(r['messages']), steps=r['steps'], calls=r['calls'],
                         bad_calls=r['bad_calls'], chars=len(ans)))
        print(f'{label} {i + 1}/{len(tasks)} · {"✓" if rows[-1]["correct"] else "✗"} · {r["steps"]} steps · '
              f'{len(ans)} chars · {time.time() - t0:.0f} s')
    json.dump(rows, open(path, 'w'), indent=1)
    return rows

def summary(results):
    print(f'{"model":<8}{"set":<11}{"tasks":>6}{"correct %":>11}{"read the file %":>17}{"finished %":>12}'
          f'{"avg steps":>11}{"avg answer chars":>18}{"bad calls":>11}')
    for label, rows in results.items():
        for s in ('synthetic', 'real'):
            rs = [r for r in rows if r['set'] == s]
            if rs:
                avg = lambda k: sum(r[k] for r in rs) / len(rs)
                print(f'{label:<8}{s:<11}{len(rs):>6}{100 * avg("correct"):>11.0f}{100 * avg("found"):>17.0f}'
                      f'{100 * avg("finished"):>12.0f}{avg("steps"):>11.1f}{avg("chars"):>18.0f}{sum(r["bad_calls"] for r in rs):>11}')

# %%
# Baseline: the untuned model on the eval tasks.
results = {'base': evaluate(eval_tasks + real_tasks, 'base')}
summary(results)

# %% [markdown]
# ## Training examples and the loss mask
# Each training task is replayed exactly as `run_agent` would run it: the tools really run on the synthetic lab, and the
# prompt for each turn is built with the same chat template. Every model turn becomes one example: the prompt
# (system prompt, tool schemas, question, earlier calls and results) gets label -100, which the loss ignores, and only
# the reply the model should write is trained on, ending with `<|im_end|>` so it also learns when to stop.

# %%
def turn_examples(task):
    set_root(task['root'])
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=task['question'])]
    out = []
    for reply, calls, train in task['trace']:
        if train:            # a context-only step (the wrong first read of a recovery trace) is never a target
            p = tok(render(messages), add_special_tokens=False)['input_ids']
            t = tok(reply + '<|im_end|>', add_special_tokens=False)['input_ids']
            out.append(dict(ids=p + t, labels=[-100] * len(p) + t))
        if calls:
            messages.append(assistant_turn(reply, calls))
            for c in calls:
                messages.append(dict(role='tool', name=c['name'], content=run_tool(c['name'], c['arguments'])))
    return out

examples = [e for t in train_tasks for e in turn_examples(t)]
skipped = sum(len(e['ids']) > MAX_LEN for e in examples)
examples = [e for e in examples if len(e['ids']) <= MAX_LEN]
n_tok = sum(len(e['ids']) for e in examples)
n_lab = sum(l != -100 for e in examples for l in e['labels'])
print(f'{len(examples)} examples ({skipped} over {MAX_LEN} tokens skipped) · {n_tok:,} tokens, of which {n_lab:,} '
      f'({100 * n_lab / n_tok:.1f}%) are trained on')
e = examples[1]
n_prompt = e['labels'].count(-100)
print('\nThe end of one example, trained part in [brackets]:\n…' + tok.decode(e['ids'][n_prompt - 40:n_prompt])
      + '[' + tok.decode(e['ids'][n_prompt:]) + ']')

# %% [markdown]
# ## LoRA
# Each targeted weight matrix W (d_out × d_in) stays frozen. LoRA adds B·A, with A of shape r × d_in and B of shape
# d_out × r, so the layer computes W·x + (α/r)·B·A·x. With r = 16 that is about 1% of the parameters. B starts at zero,
# so training starts from exactly the untuned model.

# %%
ADAPTER = f'{FT}/adapter'
if os.path.exists(f'{ADAPTER}/adapter_config.json'):
    model = PeftModel.from_pretrained(model, ADAPTER)
    print(f'loaded the trained adapter from {ADAPTER}: delete it to train again')
    losses = json.load(open(f'{FT}/losses.json'))
else:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})   # recompute activations: less memory
    model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, task_type='CAUSAL_LM',
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()       # LoRA weights in fp32: small, and fp16 gradient scaling needs fp32 parameters
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    total = math.ceil(len(examples) * EPOCHS / ACCUM)
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * s / total))))
    scaler = torch.amp.GradScaler(enabled=compute == torch.float16)
    model.train()
    losses, run, t0 = [], 0.0, time.time()
    order = random.Random(SEED)
    for epoch in range(EPOCHS):
        batch = examples[:]
        order.shuffle(batch)
        for i, e in enumerate(batch):
            x = torch.tensor([e['ids']], device=dev)
            y = torch.tensor([e['labels']], device=dev)
            with torch.autocast(dev, dtype=compute, enabled=dev == 'cuda'):
                loss = model(input_ids=x, labels=y).loss / ACCUM
            scaler.scale(loss).backward()
            run += loss.item()
            if (i + 1) % ACCUM == 0 or i + 1 == len(batch):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                losses.append(run)
                step = len(losses)
                if step % 10 == 0 or step == total:
                    el = time.time() - t0
                    print(f'step {step}/{total} · loss {sum(losses[-10:]) / len(losses[-10:]):.3f} · '
                          f'lr {sched.get_last_lr()[0]:.1e} · {el / 60:.1f} min · ETA {el / step * (total - step) / 60:.1f} min')
                run = 0.0
    model.save_pretrained(ADAPTER)
    json.dump(losses, open(f'{FT}/losses.json', 'w'))
    print(f'saved the adapter to {ADAPTER} · {(time.time() - t0) / 60:.1f} min')
model.eval()
model.config.use_cache = True

# %%
# The training loss. It should fall fast: most of each reply is format (the tool-call JSON, the answer template), and
# what remains is picking the right row and number.
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3.5))
    plt.plot(losses, lw=1, alpha=0.5, label='per step')
    k = 10
    plt.plot(range(k - 1, len(losses)), [sum(losses[i - k + 1:i + 1]) / k for i in range(k - 1, len(losses))], lw=2, label=f'{k}-step mean')
    plt.xlabel('optimizer step'); plt.ylabel('loss on the model\'s turns'); plt.yscale('log'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{FT}/loss.png', dpi=120); plt.show()
except ImportError:
    print('loss by step:', [round(l, 3) for l in losses])

# %%
# The tuned model on the same tasks, then side by side with the base model and, if it is on Drive, the v1 run. v1 had
# no sort_table and its synthetic labs had no decoys, so compare it with v2 on the real-file rows.
tuned = evaluate(eval_tasks + real_tasks, 'tuned')
v1 = f'{OUT}/finetune/eval_tuned.json'
results = {'base': results['base'], **({'v1': json.load(open(v1))} if os.path.exists(v1) and not SMOKE else {}),
           VERSION: tuned}
summary(results)

# %%
# The question the untuned agent failed in its first run (baseline_trace_qwen3-1.7b.txt). It asks two things at once,
# which no training task did, so this shows whether the habit stretches beyond the templates.
set_root(LAB)
demo = run_agent('In the recall test results, which model copies best when the first copy is far back, '
                 'and how does attn-gqa compare with llama-style?', generate, max_steps=MAX_STEPS)
print('\n══ answer ══\n' + str(demo['answer']))

# %%
# Where the tuned model still fails: its wrong answers next to the gold strings.
for r in tuned:
    if not r['correct']:
        print(f"[{r['set']}/{r['kind']}] {r['question']}\n  gold: {r['gold']}\n  got:  {r['answer'][:300]}\n")

# %% [markdown]
# ## Reading the results
# - **Correct %** is the headline. The baseline is low mostly because its answers ramble or stop early, not because it
#   misreads; *avg answer chars* shows that.
# - **Synthetic vs real.** Synthetic eval labs are new random labs of the training kind. The real tasks are about your
#   own tables, which training never saw. A large gap means the model learned the templates, not the habit: v1 scored
#   70% synthetic and 0% real. The real-file rows are the number to watch.
# - **Read the file %** (read or sorted the right table) separates two failures: never finding the table (a search
#   problem) and finding it but answering wrongly (a reading problem).
# - **v1 → v2.** If the real-file score rises, the decoys and the recovery traces fixed the search problem; if max/min
#   tasks rise, the sort tool fixed the reading problem. The list of wrong answers above shows which.
# - **The demo question** asks two things at once. Training had only one-part questions, so a tuned model may answer
#   the first part only. That is the next thing to add to the training data.
# - **What this costs.** A model trained on one narrow task can get worse at others; it may now look for a file even
#   when none is needed. Keep the untuned model for general chat, and use the adapter for lab questions.
# - To use the adapter in `mini_agent`, set `USE_ADAPTER = True` there.
