# Agent lab

The next step after the [GPT-2 architecture lab](../gpt2-lab/): build an agent on a small pretrained model whose
architecture the lab already studied, and understand every piece of it. The model is Qwen3-1.7B, the lab's
`qwen3-style` (GQA + QK-norm + RoPE) scaled up and trained on far more data than the lab could afford.

| Notebook | GPU | What it does |
|---|---|---|
| [1 · Qwen3 recall](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/agent-lab/qwen3_recall.ipynb) | L4 | The lab's recall test (copying with a control, key lookup with 1/4/16 facts) at 1K-16K tokens, at bf16, 8-bit and 4-bit. Answers: can this model keep track of a long agent context, and what does quantization cost? |
| [2 · Minimal agent](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/agent-lab/mini_agent.ipynb) | T4 or L4 | A model, four tools (`list_files`, `read_file`, `get_value`, `sort_table`) and a loop, in plain Python. It reads the lab's results on Drive and answers questions about them, printing every step. Set `USE_ADAPTER = True` to run it with the fine-tuned adapter. |
| [3 · Fine-tune the agent](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/agent-lab/agent_finetune.ipynb) | L4 | QLoRA on tool-use traces. Generates synthetic labs with known answers (tables among decoy files), trains only on the model's own turns, and scores the agent before and after on held-out synthetic tasks and on questions about the real lab files, which training never saw. About 1 hour. |

Results go to `MyDrive/agent-lab/`; the agent reads `MyDrive/gpt2-lab/`. Re-running a notebook skips finished
steps (recall precisions, evals, the trained adapter).

## Recall results

Qwen3-1.7B copies 96–99% at every length up to 16K tokens and finds single facts every time. With 16 facts it drops
to about 80% when they sit near the start of a 16K context. 4-bit costs nothing measurable (2.2 GiB against 4.1 GiB
for bf16); 8-bit is slower and no better. The model's memory is not what limits the agent.

## Fine-tuning

**v1** taught the format but not the reading. Synthetic accuracy rose from 35% to 70%, while real-file accuracy fell
from 17% to 0%. Every synthetic folder held one table, so the model learned "list, read the only file, answer". In the
real folders, full of per-run JSON files, it read the wrong file or none, and quoted file sizes as metrics. It also
got maxima wrong by comparing numbers by eye. The training loss (0.03 by step 40) showed none of this; only the
real-file eval did.

**v2** changes the data and the tools: decoy files next to every table, recovery traces where a wrong first read is
context but never a training target, and a `sort_table` tool so highest/lowest questions become a lookup. It reached
80% synthetic and 25% real, but the untuned model given the same tool also scored 25% real: the tool did more than
the training. It sent every "training results" question to `recall/`, and misread cells in the 14-column recall table.

**v3** targets those failures. 40% of synthetic labs have topics that share a word, with one table in a folder named
after that word (like `results/` beside `recall/`). A `get_value(path, run, column)` tool replaces reading tables by
eye. Every eval answer now saves its tool calls. Results go to `finetune_v3/`.

## Baseline

[baseline_trace_qwen3-1.7b.txt](baseline_trace_qwen3-1.7b.txt) is the agent's first run (Qwen3-1.7B, bf16, no
thinking). The tool calls are correct (list the folder, read the CSV), but the final reply restates the whole table,
mislabels columns and runs out of tokens before answering. That is the failure fine-tuning should fix, and this
trace is what to measure the fix against.

## Building the notebooks

The `.py` files are the source (percent-format cells); `python build_notebooks.py` regenerates the `.ipynb` files.
`agent_core.py` (tools, tool-call parsing, the loop) is pasted into both agent notebooks, so the fine-tuned model is
trained on exactly the prompt it sees when used. Without a GPU, `agent_finetune.py` runs a smoke test on a tiny random
model to check the code end to end.
