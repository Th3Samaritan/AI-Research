# The Nano Suite — four papers, one tiny Transformer, run them all

Four self-contained, heavily-commented scripts that build on your
`DIY_GPT_Dev.ipynb`. Same data (tiny-Shakespeare, char-level), same
Transformer core — each script changes exactly the thing its paper is about,
so the differences you measure are the papers' claims.

| script | paper | the one idea it isolates |
|---|---|---|
| `nanogpt_attention.py` | Vaswani 2017 / GPT | causal self-attention → generation |
| `nanobert.py` | Devlin 2018 (BERT) | delete the causal mask + mask the input → understanding |
| `nano_scaling_laws.py` | Kaplan 2020 | loss vs model size is a power law |
| `nano_chinchilla.py` | Hoffmann 2022 | at fixed compute there's an optimal N; balance N and D 50/50 |
| `compare.py` | — | reads every `runs/*.json`, prints claim-by-claim verdicts |

## Run order (CPU-friendly; add `--quick` to any script for a ~2-8 min smoke test)

```bash
python nanogpt_attention.py     # ~15-25 min  -> runs/gpt_attention.json
python nanobert.py              # ~15-25 min  -> runs/bert_mlm.json
python nano_scaling_laws.py     # ~30-45 min  -> runs/scaling_laws.json + png
python nano_chinchilla.py       # ~40-60 min  -> runs/chinchilla.json  + png
python compare.py               # instant     -> the verdict table
```

Each script downloads `input.txt` automatically on first run.

## What "comparing them" actually means (read before judging numbers)

1. **GPT vs BERT losses are different exams.** GPT predicts 100% of
   characters with left-only context; BERT predicts ~15% masked characters
   with context from both sides. BERT's lower number is expected — the
   interesting comparison is *capability*: GPT generates, BERT fills blanks.
2. **Exponents are dataset-dependent; shapes are not.** You will not get
   Kaplan's 0.076 or Chinchilla's 20 tokens/param on char-level Shakespeare.
   What must survive: a *smooth power-law-ish* L(N) curve (Kaplan) and
   *IsoFLOP valleys* with N_opt ∝ C^a, a nearer 0.5 than 0.73 (Chinchilla).
3. **The two scaling papers disagree, and that's the point.** Kaplan: grow
   the model (N ∝ C^0.73). Chinchilla: grow both equally (N ∝ C^0.50) —
   Kaplan's fixed learning-rate schedule under-credited the long-data runs.
   `nano_chinchilla.py` implements the fix (cosine schedule matched to each
   run's horizon) so you can see which rule your own valleys obey.

## Companion breakdowns (Paper To Code, house style)

- `../Paper To Code/scaling law/` — Kaplan: animated site + math PDF (done)
- `../Paper To Code/chincillas/`  — Chinchilla: animated site + math PDF
- `../Paper To Code/bert/`        — BERT: animated site + math PDF
