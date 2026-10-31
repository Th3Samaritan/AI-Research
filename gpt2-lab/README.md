# GPT-2 (124M) architecture lab

Train GPT-2 124M from scratch on Colab (A100 or L4), train same-size models with the modern changes from the
[LLM Inference Field Guide](https://th3samaritan.github.io/AI-Research/llm-inference/) (RoPE, RMSNorm, SwiGLU, QK-norm,
GQA/MQA), and compare them all on one benchmark.

**[Open the notebook in Colab](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/gpt2_arch_lab.ipynb)**

This is the real-scale step up from the [Nano Suite](../Nanogpt/) (character-level Shakespeare on a CPU): real GPT-2 size,
real web text, a real GPU, and the benchmarks people actually report.

`gpt2_arch_lab.py` is the same code as a script. It runs on any GPU machine with `python gpt2_arch_lab.py`.

## What gets compared

Every run uses the same data (FineWeb, pre-tokenized with GPT-2's tokenizer), the same token order, the same init seed,
the same optimizer and learning-rate schedule. Only the architecture changes.

| Variant | Change from GPT-2 | Params | KV cache / token |
|---|---|---|---|
| `gpt2` | baseline: LayerNorm, learned positions, GELU MLP, biases, 12 heads | 124.5M | 36 KiB |
| `gpt2+rope` | rotary position embeddings instead of learned positions | 123.7M | 36 KiB |
| `gpt2+rmsnorm` | RMSNorm instead of LayerNorm | 124.5M | 36 KiB |
| `gpt2+swiglu` | SwiGLU MLP (8/3·d wide, same parameter count) | 124.5M | 36 KiB |
| `gpt2+relu2` | squared-ReLU MLP | 124.5M | 36 KiB |
| `llama-style` | RoPE + RMSNorm + SwiGLU, no biases | 123.6M | 36 KiB |
| `qwen3-style` | llama-style + QK-norm + GQA (12 query / 4 KV heads), MLP widened to keep ~124M | 124.8M | 12 KiB |
| `qwen3-mqa` | as above with 1 KV head | 124.8M | 3 KiB |

Each `gpt2+…` row changes one thing, so a difference in loss is caused by that change.

## The benchmark

| Metric | Meaning |
|---|---|
| **Validation loss** (and perplexity) on 10.5M held-out FineWeb tokens | The main score. Lower is better |
| **Loss vs tokens** curve | Sample efficiency: which architecture learns more per token |
| **Loss vs wall-clock** curve | Cost efficiency: does the change pay for any slowdown it causes |
| **HellaSwag** acc_norm, 10,042 questions, zero-shot | Common-sense completion. Random is 25%; OpenAI's GPT-2 124M scores about 29.5% |
| Training **tokens/s**, **MFU**, peak memory | Training cost on your GPU |
| **KV-cache bytes/token**, prefill and decode tokens/s at batch 1 and 16 | Inference cost, which is where GQA and MQA pay off |

OpenAI's released GPT-2 124M is scored on the same benchmark as a reference row.

How to read the numbers:
- Two seeds of the same model differ by about 0.01–0.02 in val loss at this scale, so smaller gaps are ties.
- HellaSwag's standard error is about ±0.45 points, so gaps under ~1.5 points don't mean much.
- Short runs score 25–28% on HellaSwag for every variant. Val loss separates them much earlier.

## Budgets and time

These times are estimates from FLOPs per token at typical utilization. The notebook prints the real tokens/s within the
first minute, so you can adjust.

| `BUDGET` | Tokens | A100 (≈120–170k tok/s) | L4 (≈35–55k tok/s) | Use it for |
|---|---|---|---|---|
| `smoke` | 20M | 2–3 min | 6–10 min | checking the pipeline works |
| `quick` | 100M | 10–14 min | 30–50 min | learning-rate sweeps, first look |
| `standard` | 500M | 50–70 min | 2.5–4 h | the comparison runs |
| `long` | 2.5B (~20 tokens/param) | 4–6 h | 13–20 h | the final model of the winning variant |

Add about 1 minute of compilation and 2–5 minutes of benchmarking per run. Checkpoints go to Google Drive every 10% of a
run, so running all again after a disconnect resumes where it stopped.

## Suggested plan

1. `BUDGET='smoke'`, `RUNS=['gpt2']`: checks the whole pipeline in a few minutes.
2. `BUDGET='quick'`: sweep `LR` over 6e-4, 1e-3 and 2e-3 for `gpt2` and `qwen3-style`. Modern variants usually tolerate
   higher learning rates, so compare each at its own best.
3. `BUDGET='standard'`: `RUNS=['gpt2', 'qwen3-style']` at those learning rates.
4. The single-change ablations (`gpt2+rope`, `gpt2+rmsnorm`, `gpt2+swiglu`) to see which change contributes what.
5. `BUDGET='long'` for the winner.

All results go into `MyDrive/gpt2-lab/results/` as one JSON file per run, plus `summary.csv` and `comparison.png`.

## Credits

The model and training recipe follow Karpathy's nanoGPT / build-nanogpt; the pre-tokenized FineWeb shards are the ones
modded-nanogpt uses (`kjj0/fineweb10B-gpt2`). HellaSwag is loaded from the Hugging Face copy (`Rowan/hellaswag`), because
the original GitHub repository has been blocked since 2026.
