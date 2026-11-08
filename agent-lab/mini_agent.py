# %% [markdown]
# # A minimal agent: a model, a few tools and a loop
#
# An agent is a language model that can act, not only answer. This notebook builds the smallest one that does
# something real: it reads your GPT-2 lab results on Drive and answers questions about them. Every piece is here in
# plain Python, so you can see what each one does and why it exists.
#
# | Piece | What it is | Why it's needed |
# |---|---|---|
# | Model | Qwen3-1.7B, instruction-tuned | Decides the next step: answer, or call a tool. Its architecture is the lab's `qwen3-style`, scaled up |
# | Tools | plain Python functions: `list_files`, `read_file`, `get_value`, `sort_table` | The model can only produce text. Tools let that text cause actions and bring back facts it doesn't know. The last two do the table work a 1.7B model gets wrong by eye |
# | Tool schema | each tool's name, description and arguments, put in the prompt | The model can only call what it has been told exists, with the arguments it has been told it takes |
# | Loop | generate → if it asked for a tool, run it, append the result, generate again | One generation can't both ask for a file and read it. The loop turns single replies into multi-step work |
# | Memory | the message list, which becomes the context window | Everything the agent knows lives here. This is what the recall test measured |
#
# The trace prints every step: what the model wrote, which tool ran, what came back, and how full the context is.
#
# **On Colab:** Runtime → Change runtime type → **T4** or **L4** → Run all. The model needs about 4 GB of GPU memory.
# To use the fine-tuned model from `agent_finetune`, set `USE_ADAPTER = True` in the config cell.

# %%
# Config.
import os, sys
IN_COLAB = 'google.colab' in sys.modules
MODEL = os.environ.get('MODEL', 'Qwen/Qwen3-1.7B')
LAB = os.environ.get('AGENT_ROOT', '/content/drive/MyDrive/gpt2-lab' if IN_COLAB else '.')   # the only folder tools can see
OUT = os.environ.get('AGENT_OUT', '/content/drive/MyDrive/agent-lab' if IN_COLAB else 'out')
USE_ADAPTER = False      # True: load the LoRA adapter trained by agent_finetune
ADAPTER_VERSION = 'v4'   # which run's adapter: 'v4' loads OUT/finetune_v4/adapter, 'v1' OUT/finetune/adapter
MAX_STEPS = 8            # model replies before the loop gives up: an agent that never stops is the classic failure
MAX_NEW_TOKENS = 768     # per model reply
THINKING = False         # Qwen3 can reason in a <think> block before replying: better plans, many more tokens

# %%
# Setup: the model and its tokenizer.
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if dev == 'cpu' or torch.cuda.get_device_capability()[0] >= 8 else torch.float16   # T4: fp16
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map=dev if dev == 'cuda' else None).eval()
if USE_ADAPTER:
    if IN_COLAB:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'peft'], check=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, f'{OUT}/finetune{"" if ADAPTER_VERSION == "v1" else "_" + ADAPTER_VERSION}/adapter').eval()
print(f'{MODEL}{" + LoRA adapter" if USE_ADAPTER else ""} on {dev} · '
      f'{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters')

# %%
from agent_core import *   # build_notebooks.py pastes agent_core.py here

# %%
set_root(LAB)

@torch.no_grad()
def generate(messages):
    prompt = tok.apply_chat_template(messages, tools=list(TOOLS.values()), add_generation_prompt=True,
                                     tokenize=False, enable_thinking=THINKING)
    ids = tok(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    reply = tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True)
    return reply, ids['input_ids'].shape[1]

# %%
# Try it. Change the task to anything about the files in LAB.
result = run_agent('In the recall test results, which model copies best when the first copy is far back, '
                   'and how does attn-gqa compare with llama-style?', generate, max_steps=MAX_STEPS)
print('\n══ answer ══\n' + str(result['answer']))

# %% [markdown]
# ## What to look for in the trace
# - **The first step is a tool call, not an answer.** The system prompt and the tool schemas steered it to look before
#   answering. Remove the tools (`TOOLS.clear()`) and it will answer from memory: fluent and wrong.
# - **Context grows every step.** Each tool result is appended and stays. At 4,000 characters per read that's about
#   1,000 tokens per file, so a 20-step task can reach 16K+ tokens: the range the Qwen3 recall test checks.
# - **Errors are fed back, not raised.** Ask it to read a file that doesn't exist and watch it recover by listing the
#   folder first.
# - **Where a 1.7B model breaks:** malformed tool calls, re-reading the same file, or misreading a table. The untuned
#   model's first run (baseline_trace_qwen3-1.7b.txt) called the tools correctly, then restated the whole table instead
#   of answering. `agent_finetune` trains that out and measures the change.
