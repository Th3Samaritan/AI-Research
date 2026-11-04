# %%
# Recall test settings. Every model answers exactly the same questions (fixed seed).
CTX = 1024                                   # every test fills the context, so the question is always at the same position
COPY_LEN = 32                                # tokens in the sequence to copy
COPY_GAPS = (0, 64, 192, 256, 384, 512, 768, 896)   # filler tokens between the two copies
COPY_TRIALS = 128                            # per gap
KV_FACTS = (1, 4, 16)                        # facts in the prompt: memory load
KV_GAPS = (32, 256, 768)                     # filler tokens between the facts and the question
KV_TRIALS = 300                              # per condition: ±3 points at 50% accuracy
KV_CHOICES = 8                               # answers to choose from: chance = 12.5%
BATCH = 8
RECALL_SEED = 2026

# %%
# Build the test cases from held-out FineWeb text.
import random
NAMES = '''Maria James Sarah David Anna John Emma Peter Laura Michael Alice Robert Julia Thomas Grace Daniel Helen Mark
Linda Paul Karen Steven Susan Kevin Nancy Brian Lisa Jason Amy Eric Kate Adam Rachel Ryan Jessica Andrew Emily Scott
Claire Henry Diana Oliver Sophie Martin Rose Victor Nina Simon'''.split()
NAMES = [n for n in NAMES if len(enc.encode(' ' + n)) == 1]            # single-token names
VALUES = [str(v) for v in range(100, 1000) if len(enc.encode(f' {v}')) == 1]   # single-token numbers: one logit per answer

def build_cases(val):
    rng = random.Random(RECALL_SEED)
    counts = np.bincount(val.astype(np.int64), minlength=enc.n_vocab)
    counts[enc.eot_token] = 0
    common = np.argsort(-counts)[:20000].tolist()                      # ordinary tokens, in random order: nothing to guess
    def filler(n):
        s = rng.randrange(len(val) - n - 1)
        return val[s:s + n].astype(np.int64).tolist()
    copy = {}
    for gap in COPY_GAPS:
        if gap + 2 * COPY_LEN > CTX:
            continue
        copy[gap] = []
        for _ in range(COPY_TRIALS):
            span, other = rng.sample(common, COPY_LEN), rng.sample(common, COPY_LEN)
            pre, mid = filler(CTX - 2 * COPY_LEN - gap), filler(gap)
            copy[gap].append((pre + span + mid + span, pre + other + mid + span))   # (test, control: different first copy)
    def fact(n, v):
        return enc.encode(f' The code for {n} is {v}.')
    def kv_case(n_facts, gap):
        names, vals = rng.sample(NAMES, max(n_facts, 1)), rng.sample(VALUES, KV_CHOICES + n_facts)
        target = rng.randrange(max(n_facts, 1))
        facts = sum((fact(names[i], vals[i]) for i in range(n_facts)), [])
        query = enc.encode(f' The code for {names[target]} is')
        # wrong answers: the other numbers in this prompt first (so the model has to use the name), then fresh ones
        wrong = [vals[i] for i in range(n_facts) if i != target][:KV_CHOICES - 1]
        wrong += vals[n_facts:n_facts + KV_CHOICES - 1 - len(wrong)]
        answer = vals[target]
        choices = [enc.encode(f' {v}')[0] for v in [answer] + wrong]
        pre = filler(CTX - len(facts) - gap - len(query))
        return pre + facts + filler(gap) + query, choices
    kv = {}
    for n in KV_FACTS:
        for gap in KV_GAPS:
            if gap + n * 12 + 16 < CTX:
                kv[f'{n}|{gap}'] = [kv_case(n, gap) for _ in range(KV_TRIALS)]
    kv['control'] = [kv_case(0, KV_GAPS[len(KV_GAPS) // 2]) for _ in range(KV_TRIALS)]   # no fact in the prompt: must be chance
    return dict(copy=copy, kv=kv)

_, val_file = get_data(0)                                              # validation shard: text no model trained on
CASES = build_cases(load_shard(val_file)[:20_000_000])
print(f"{len(NAMES)} names, {len(VALUES)} answer values, {sum(map(len, CASES['copy'].values()))} copy trials, "
      f"{sum(map(len, CASES['kv'].values()))} lookup trials")

# %%
# Run the tests on one model.
@torch.no_grad()
def copy_score(logits_fn, seqs):
    """Share of the second copy's tokens predicted exactly (top-1), skipping the first 2 tokens, which only cue the match."""
    L, T, right = COPY_LEN, len(seqs[0]), 0.0
    for i in range(0, len(seqs), BATCH):
        x = torch.tensor(seqs[i:i + BATCH], device=dev)
        with autocast():
            lg = logits_fn(x)[:, T - L + 1:T - 1]
        right += (lg.argmax(-1) == x[:, T - L + 2:]).float().mean(1).sum().item()
    return right / len(seqs)

@torch.no_grad()
def kv_score(logits_fn, cases):
    """Share of questions where the correct number gets the highest probability among the choices (the first choice)."""
    right = 0
    for i in range(0, len(cases), BATCH):
        chunk = cases[i:i + BATCH]
        x = torch.tensor([c[0] for c in chunk], device=dev)
        ch = torch.tensor([c[1] for c in chunk], device=dev)
        with autocast():
            last = logits_fn(x)[:, -1].float()
        right += (last.gather(1, ch).argmax(1) == 0).sum().item()
    return right / len(cases)

def recall_test(logits_fn):
    r = dict(copy={}, copy_control={}, kv={})
    for gap, pairs in CASES['copy'].items():
        r['copy'][str(gap)] = copy_score(logits_fn, [p[0] for p in pairs])
        r['copy_control'][str(gap)] = copy_score(logits_fn, [p[1] for p in pairs])
    for key, cases in CASES['kv'].items():
        r['kv'][key] = kv_score(logits_fn, cases)
    return r

# %%
# Test every trained model on Drive, then OpenAI's GPT-2. Finished models are skipped, so Run all again resumes.
os.makedirs(f'{OUT}/recall', exist_ok=True)
for p in sorted(glob.glob(f'{OUT}/ckpt/*__{BUDGET}*_final.pt')):
    tag = os.path.basename(p)[:-len('_final.pt')]
    path = f'{OUT}/recall/{tag}.json'
    if os.path.exists(path):
        print(f'{tag}: already tested'); continue
    t0 = time.time()
    m = load_final(tag).eval()
    r = recall_test(lambda i: m(i)[0])
    json.dump(dict(tag=tag, name=tag.split('__')[0], **r), open(path, 'w'), indent=1)
    print(f"{tag}: copy near {100*r['copy'].get('64', 0):.0f}% · far {100*r['copy'].get('768', 0):.0f}% · {time.time()-t0:.0f} s")
    del m; gc.collect()
    if dev == 'cuda': torch.cuda.empty_cache()
path = f'{OUT}/recall/openai-gpt2.json'
if not os.path.exists(path):
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained('gpt2').to(dev).eval()
    json.dump(dict(tag='openai-gpt2', name='openai-gpt2', **recall_test(lambda i: hf(i).logits)), open(path, 'w'), indent=1)
    del hf; gc.collect()

# %%
# Results: the table, then the charts.
import pandas as pd
import matplotlib.pyplot as plt
R = [json.load(open(p)) for p in sorted(glob.glob(f'{OUT}/recall/*.json'))]
def mean(d, keys):
    v = [d[str(k)] for k in keys if str(k) in d]
    return 100 * sum(v) / len(v) if v else float('nan')
near = [g for g in COPY_GAPS if g + COPY_LEN <= 256]; far = [g for g in COPY_GAPS if g >= 384]
rows = []
for r in R:
    row = {'run': r['name'], 'copy near %': round(mean(r['copy'], near), 1), 'copy far %': round(mean(r['copy'], far), 1),
           'control %': round(mean(r['copy_control'], COPY_GAPS), 1)}
    for k, v in r['kv'].items():
        row['no fact %' if k == 'control' else '{} fact{} @{}'.format(k.split('|')[0], '' if k.startswith('1|') else 's', k.split('|')[1])] = round(100 * v, 1)
    rows.append(row)
table = pd.DataFrame(rows).sort_values('copy far %', ascending=False)
display(table) if 'display' in globals() else print(table.to_string(index=False))
table.to_csv(f'{OUT}/recall/summary.csv', index=False)

fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
for r in R:
    gaps = sorted(int(g) for g in r['copy'])
    ax[0].plot(gaps, [100 * r['copy'][str(g)] for g in gaps], marker='o', ms=3, lw=2.5 if r['name'] == 'openai-gpt2' else 1.3, label=r['name'])
ax[0].axvline(256 - COPY_LEN, c='gray', ls=':', label='sliding window ends')
ax[0].set(xlabel='filler tokens between the two copies', ylabel='% of tokens copied correctly', title='Copying across a gap', ylim=(0, 100))
ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)
keys = [k for k in R[0]['kv'] if k != 'control']
order = sorted(R, key=lambda r: -mean(r['kv'], keys))
im = ax[1].imshow([[100 * r['kv'].get(k, float('nan')) for k in keys] for r in order], aspect='auto', cmap='viridis', vmin=0, vmax=100)
ax[1].set_xticks(range(len(keys)), [k.replace('|', ' facts, gap ') for k in keys], rotation=40, ha='right', fontsize=8)
ax[1].set_yticks(range(len(order)), [r['name'] for r in order], fontsize=8)
for i, r in enumerate(order):
    for j, k in enumerate(keys):
        if k in r['kv']:
            ax[1].text(j, i, f"{100*r['kv'][k]:.0f}", ha='center', va='center', fontsize=7, color='white' if r['kv'][k] < .6 else 'black')
ax[1].set_title(f'Key lookup accuracy, % (chance {100/KV_CHOICES:.1f}%)')
plt.tight_layout(); plt.savefig(f'{OUT}/recall/recall.png', dpi=120); plt.show()
