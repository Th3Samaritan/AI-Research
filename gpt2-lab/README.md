# GPT-2 (124M) architecture lab

Train GPT-2 124M from scratch on Colab (A100 or L4), train same-size models with the modern changes and the attention mechanisms from the
[LLM Inference Field Guide](https://th3samaritan.github.io/AI-Research/llm-inference/) (RoPE, RMSNorm, SwiGLU, QK-norm; MHA,
GQA, MQA, MLA, sliding window, DSA, Gated DeltaNet), compare them all on one benchmark, then download and use them offline.

This is the real-scale step up from the [Nano Suite](../Nanogpt/) (character-level Shakespeare on a CPU): real GPT-2 size,
real web text, a real GPU, and the benchmarks people actually report.

## The notebooks

Two families of notebooks: one notebook per **architecture** (GPT-2 and the modern changes to it) and one per
**attention mechanism** (MHA, GQA, MQA, MLA, sliding window, DSA, Gated DeltaNet and hybrids), plus a compare
notebook, an all-in-one notebook and an offline one. They all share
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
| [gpt2_arch_lab](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/gpt2_arch_lab.ipynb) | all-in-one: every architecture and every attention mechanism (16 runs) by default; finished runs are skipped, so Run all again continues where it stopped |
| [use_offline](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/use_offline.ipynb) | run a downloaded model on your own machine |

Each variant notebook opens with what its change does, why, and what to look for in the results.

### The attention family

Same base for all of them (the llama-style model: RoPE, RMSNorm, SwiGLU, no biases, ~124M parameters), and only the
attention mechanism changes. Each notebook explains how its mechanism works, what it saves and what it costs.

| Notebook | Mechanism | Used in | Cache @1K ctx | Decode intensity b1 / b16 / b128 |
|---|---|---|---|---|
| [1 · MHA](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/attention/1_mha.ipynb) | every head has its own K and V (the reference) | GPT-2, Llama 1 | 36 MiB | 1.0 / 5.4 / 7.2 |
| [2 · GQA](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/attention/2_gqa.ipynb) | 12 query heads share 4 K/V heads | Llama 3, Qwen, Mistral | 12 MiB | 1.1 / 10.1 / 19.5 |
| [3 · MQA](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/attention/3_mqa.ipynb) | all query heads share 1 K/V head | PaLM, Falcon | 3 MiB | 1.1 / 15.3 / 56.0 |
| [4 · MLA](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/attention/4_mla.ipynb) | K/V rebuilt from a 256-dim cached latent + decoupled RoPE | DeepSeek V2/V3 | 6.75 MiB | 1.6 / 18.1 / 45.3 |
| [5 · SWA](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/attention/5_swa.ipynb) | each query sees the last 256 tokens | Mistral 7B | 9 MiB (capped) | 1.0 / 10.3 / 22.6 |
| [6 · SWA + global](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/attention/6_swa_global.ipynb) | 5 sliding-window layers : 1 full | Gemma 3 | 13.5 MiB | 1.0 / 8.8 / 16.2 |
| [7 · DSA](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/attention/7_dsa.ipynb) | lightning indexer picks the top 256 keys per query | DeepSeek V3.2 | 36.75 MiB (reads 256 keys) | 1.0 / 10.1 / 21.3 |
| [8 · Gated DeltaNet](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/attention/8_gdn.ipynb) | linear attention: fixed 128×128 state per head, delta rule | Qwen3-Next, Kimi Linear (linear layers) | 2.25 MiB (constant) | 1.0 / 14.3 / 59.5 |
| [9 · GDN hybrid](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/attention/9_gdn_hybrid.ipynb) | 3 Gated DeltaNet layers : 1 full attention | Qwen3-Next | 10.7 MiB | 1.0 / 9.9 / 20.0 |

All of them pass the same check: decoding token by token with their own cache (K/V, latent, window, top-k selection or
fixed state) gives exactly the same outputs as running the whole sequence at once.

Speed caveats: sliding window and DSA apply their pattern as a mask on a dense attention kernel, so their training
tokens/s show no saving (DSA is slower, since it also computes the full attention pattern to train its indexer). The
cache and intensity columns show what optimized kernels gain. Gated DeltaNet uses the fla-core Triton kernels on GPU
(installed automatically), checked against a plain PyTorch recurrence before training starts. MLA trains with K/V
rebuilt explicitly; its intensity numbers assume the absorbed form inference engines use.

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

Arithmetic intensity (FLOPs per byte from GPU memory) of one inference step at 1,024 tokens of context. The A100's ridge
point is about 150-200 and the L4's about 400: below it a step is memory-bound.

| Variant | decode batch 1 | batch 16 | batch 128 | prefill 1K |
|---|---|---|---|---|
| `gpt2` (and the single-change variants, llama-style) | 1.0 | 5.3 | 7.2 | 950 |
| `qwen3-style` (GQA, 4 KV heads) | 1.1 | 10.2 | 19.8 | 1,048 |
| `qwen3-mqa` (1 KV head) | 1.1 | 15.3 | 56.3 | 1,088 |

Batching raises intensity because the weights are read once for every sequence in the batch, but the KV cache grows
with the batch, so multi-head attention levels off near 7. Fewer KV heads lift that ceiling.

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
| **Arithmetic intensity** (FLOPs per byte) of decode at batch 1/16/128 and of prefill, plus achieved GB/s and a roofline chart for your GPU | Whether each step is memory- or compute-bound, and how far GQA/MQA lift the batched-decode ceiling |

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

## Running everything, across several sessions

The [all-in-one notebook](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/gpt2-lab/gpt2_arch_lab.ipynb)
trains `RUNS = ARCHITECTURES + ATTENTIONS` by default: all 16 models, in order. A run with results already on Drive
is skipped without retraining or re-benchmarking, an interrupted run resumes from its last checkpoint, and an
exported model isn't exported again. So when Colab disconnects, open the notebook and Run all again; it picks up at the
next unfinished run. Runs made in the separate notebooks count too (same Drive folder). To train only some, set e.g.
`RUNS = ATTENTIONS` or `RUNS = ['attn-mla', 'attn-dsa']`.

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
