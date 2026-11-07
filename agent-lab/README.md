# Agent lab

The next step after the [GPT-2 architecture lab](../gpt2-lab/): build an agent on a small pretrained model whose
architecture the lab already studied, and understand every piece of it. The model is Qwen3-1.7B, the lab's
`qwen3-style` (GQA + QK-norm + RoPE) scaled up and trained on far more data than the lab could afford.

| Notebook | GPU | What it does |
|---|---|---|
| [1 · Qwen3 recall](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/agent-lab/qwen3_recall.ipynb) | L4 | The lab's recall test (copying with a control, key lookup with 1/4/16 facts) at 1K-16K tokens, at bf16, 8-bit and 4-bit. Answers: can this model keep track of a long agent context, and what does quantization cost? |
| [2 · Minimal agent](https://colab.research.google.com/github/Th3Samaritan/AI-Research/blob/master/agent-lab/mini_agent.ipynb) | T4 or L4 | A model, two tools (`list_files`, `read_file`) and a loop, in plain Python. It reads the lab's results on Drive and answers questions about them, printing every step. |

Results go to `MyDrive/agent-lab/`; the agent reads `MyDrive/gpt2-lab/`. Re-running the recall notebook skips
finished precisions.

## Baseline

[baseline_trace_qwen3-1.7b.txt](baseline_trace_qwen3-1.7b.txt) is the agent's first run (Qwen3-1.7B, bf16, no
thinking). The tool calls are correct (list the folder, read the CSV), but the final reply restates the whole table,
mislabels columns and runs out of tokens before answering. That is the failure fine-tuning should fix, and this
trace is what to measure the fix against.

## Building the notebooks

The `.py` files are the source (percent-format cells); `python build_notebooks.py` regenerates the `.ipynb` files.
