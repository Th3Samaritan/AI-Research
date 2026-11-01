# GPT-2 (124M) architecture lab

Train GPT-2 124M from scratch on Colab (A100 or L4), train same-size models with the modern changes from the
[LLM Inference Field Guide](https://th3samaritan.github.io/AI-Research/llm-inference/) (RoPE, RMSNorm, SwiGLU, QK-norm,
GQA/MQA), compare them all on one benchmark, then download the models and use them offline.

This is the real-scale step up from the [Nano Suite](../Nanogpt/) (character-level Shakespeare on a CPU): real GPT-2 size,
real web text, a real GPU, and the benchmarks people actually report.

## The notebooks

One notebook per architecture, plus a compare notebook, an all-in-one notebook and an offline one. They all share
`MyDrive/gpt2-lab/` on Google Drive, so runs from different notebooks and sessions land in one comparison.

| Notebook | What it does |
|---|---|
| [1 · gpt2](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/variants/1_gpt2.ipynb) | the 2019 baseline everything is measured against |
| [2 · gpt2+rope](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/variants/2_gpt2_rope.ipynb) | rotary positions instead of a learned position table |
| [3 · gpt2+rmsnorm](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/variants/3_gpt2_rmsnorm.ipynb) | RMSNorm instead of LayerNorm |
| [4 · gpt2+swiglu](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/variants/4_gpt2_swiglu.ipynb) | SwiGLU MLP at the same parameter count |
| [5 · gpt2+relu2](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/variants/5_gpt2_relu2.ipynb) | squared-ReLU MLP |
| [6 · llama-style](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/variants/6_llama-style.ipynb) | RoPE + RMSNorm + SwiGLU, no biases |
| [7 · qwen3-style](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/variants/7_qwen3-style.ipynb) | + QK-norm + GQA (3× smaller KV cache) |
| [8 · qwen3-mqa](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/variants/8_qwen3-mqa.ipynb) | + a single KV head (12× smaller KV cache) |
| [compare_results](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/compare_results.ipynb) | table, curves, OpenAI GPT-2 reference row, text from every model |
| [gpt2_arch_lab](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/gpt2_arch_lab.ipynb) | all-in-one: train any list of variants, compare, export |
| [use_offline](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/use_offline.ipynb) | run a downloaded model on your own machine |

Each variant notebook opens with what its change does, why, and what to look for in the results.

`gpt2_arch_lab.py` is the source of all of them (`python build_notebooks.py` regenerates the notebooks), and it also
runs as a plain script on any cloud GPU machine.

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
| **HellaSwag 0 / 1 / 5-shot** on the same 2,000 questions (`SHOTS`, `FEWSHOT_N`) | Whether solved examples in the prompt help (in-context learning). At 124M expect little or no gain; the GPT-3 paper saw the same for its 125M model |
| Training **tokens/s**, **MFU**, peak memory | Training cost on your GPU |
| **KV-cache bytes/token**, prefill and decode tokens/s at batch 1 and 16 | Inference cost, which is where GQA and MQA pay off |

OpenAI's released GPT-2 124M is scored on the same benchmark as a reference row (in the compare notebook).

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

Add about 1 minute of compilation and 5–10 minutes of benchmarking per run (the 5-shot prompts are about 6× longer). Checkpoints go to Google Drive every 10% of a
run, so running all again after a disconnect resumes where it stopped.

## Test it before a long run

1. Open the [gpt2 notebook](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/variants/1_gpt2.ipynb),
   set `BUDGET = 'smoke'`, and Run all (a few minutes on an A100).
2. Check: tokens/s in the expected range, val loss falling from about 10.8 to about 6–7, a results row in the table,
   and an export zip in `MyDrive/gpt2-lab/export/`.
3. Download that zip and run it offline (next section). If all of that works, the long runs will too.

The code itself was smoke-tested on CPU with a tiny model: every variant trains, benchmarks, exports and reloads, and
the KV-cache decoding gives the same outputs as the full forward pass.

## Download your model and use it offline

When a run finishes, the notebook exports it to `MyDrive/gpt2-lab/export/<run>/` and zips it:

| File | Size | Use it with |
|---|---|---|
| `<run>_fp16.pt` | ~250 MB | `use_offline.py`: torch + tiktoken only, CPU or GPU |
| `hf/` (gpt2, llama-style, qwen3-style, qwen3-mqa) | ~500 MB | Hugging Face Transformers, vLLM, llama.cpp (GGUF + quantization) |

The `hf/` folder is checked at export time: the Transformers model must give the same outputs as the trained one.

To download: Google Drive → `MyDrive/gpt2-lab/export/` → right-click `<run>.zip` → Download. Unzip it next to
[use_offline.py](use_offline.py), then:

```
pip install torch tiktoken
python use_offline.py gpt2__standard/gpt2__standard_fp16.pt --prompt "The history of the printing press"
python use_offline.py gpt2__standard/gpt2__standard_fp16.pt --chat          # prompts in a loop
python use_offline.py gpt2__standard/gpt2__standard_fp16.pt --ppl essay.txt # perplexity on your own text
python use_offline.py gpt2__standard/gpt2__standard_fp16.pt --bench         # speed on this machine
```

A 124M model runs comfortably on a laptop CPU. It's a base model: it continues text rather than following instructions.
[use_offline.ipynb](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/use_offline.ipynb)
shows the same in a notebook, plus the Transformers, vLLM and llama.cpp routes.

## Suggested plan

1. Smoke-test the gpt2 notebook (above).
2. `BUDGET='quick'`: sweep `LR` over 6e-4, 1e-3 and 2e-3 for `gpt2` and `qwen3-style`. Modern variants usually tolerate
   higher learning rates, so compare each at its own best. A non-default LR gets its own result file (`…__lr0.002`).
3. `BUDGET='standard'`: the gpt2 and qwen3-style notebooks at those learning rates.
4. The single-change notebooks (2–5) to see which change contributes what, and llama-style to see if the gains add up.
5. `BUDGET='long'` for the winner, then export it and quantize it.

All results go into `MyDrive/gpt2-lab/results/` as one JSON file per run, plus `summary.csv` and `comparison.png`.

## Credits

The model and training recipe follow Karpathy's nanoGPT / build-nanogpt; the pre-tokenized FineWeb shards are the ones
modded-nanogpt uses (`kjj0/fineweb10B-gpt2`). HellaSwag is loaded from the Hugging Face copy (`Rowan/hellaswag`), because
the original GitHub repository has been blocked since 2026.
