# %% [markdown]
# # A minimal agent: a model, two tools and a loop
#
# An agent is a language model that can act, not only answer. This notebook builds the smallest one that does
# something real: it reads your GPT-2 lab results on Drive and answers questions about them. Every piece is here in
# plain Python, so you can see what each one does and why it exists.
#
# | Piece | What it is | Why it's needed |
# |---|---|---|
# | Model | Qwen3-1.7B, instruction-tuned | Decides the next step: answer, or call a tool. Its architecture is the lab's `qwen3-style`, scaled up |
# | Tools | plain Python functions: `list_files`, `read_file` | The model can only produce text. Tools let that text cause actions and bring back facts it doesn't know |
# | Tool schema | each tool's name, description and arguments, put in the prompt | The model can only call what it has been told exists, with the arguments it has been told it takes |
# | Loop | generate → if it asked for a tool, run it, append the result, generate again | One generation can't both ask for a file and read it. The loop turns single replies into multi-step work |
# | Memory | the message list, which becomes the context window | Everything the agent knows lives here. This is what the recall test measured |
#
# The trace prints every step: what the model wrote, which tool ran, what came back, and how full the context is.
#
# **On Colab:** Runtime → Change runtime type → **T4** or **L4** → Run all. The model needs about 4 GB of GPU memory.

# %%
# Setup: the model and its tokenizer.
import os, sys, re, json, subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
MODEL = os.environ.get('MODEL', 'Qwen/Qwen3-1.7B')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if dev == 'cpu' or torch.cuda.get_device_capability()[0] >= 8 else torch.float16   # T4: fp16
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map=dev if dev == 'cuda' else None).eval()
print(f'{MODEL} on {dev} · {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters')

# %%
# Config.
ROOT = os.environ.get('AGENT_ROOT', '/content/drive/MyDrive/gpt2-lab' if IN_COLAB else '.')   # the only folder tools can see
MAX_STEPS = 8            # tool calls before the loop gives up: an agent that never stops is the classic failure
MAX_NEW_TOKENS = 768     # per model reply
THINKING = False         # Qwen3 can reason in a <think> block before replying: better plans, many more tokens
READ_LIMIT = 4000        # characters a single read_file returns: every result stays in the context for good

# %%
# Tools. Plain Python functions. The type hints and the docstring are not decoration: transformers turns them into the
# JSON schema that goes into the prompt, and that schema is all the model knows about each tool.
def safe_path(path: str) -> str:
    """Keep every tool inside ROOT, so the model can't read anything else on the machine."""
    full = os.path.realpath(os.path.join(ROOT, path))
    if os.path.commonpath([full, os.path.realpath(ROOT)]) != os.path.realpath(ROOT):
        raise ValueError(f'{path} is outside the allowed folder')
    return full

def list_files(folder: str = '.') -> str:
    """
    List the files and subfolders in a folder, with file sizes.

    Args:
        folder: Path relative to the lab folder, for example "." or "recall".
    """
    full = safe_path(folder)
    out = []
    for name in sorted(os.listdir(full)):
        p = os.path.join(full, name)
        out.append(f'{name}/' if os.path.isdir(p) else f'{name}  ({os.path.getsize(p):,} bytes)')
    return '\n'.join(out) or '(empty folder)'

def read_file(path: str) -> str:
    """
    Read a text file (csv, json, md, txt, py). Long files are cut off.

    Args:
        path: Path relative to the lab folder, for example "recall/summary.csv".
    """
    text = open(safe_path(path), encoding='utf-8', errors='replace').read()
    return text if len(text) <= READ_LIMIT else text[:READ_LIMIT] + f'\n... [cut: {len(text):,} characters in total]'

TOOLS = {f.__name__: f for f in (list_files, read_file)}

def run_tool(name, args):
    """Run one tool call. Errors go back to the model as text, so it can correct itself instead of crashing the loop."""
    if name not in TOOLS:
        return f'Error: there is no tool named {name}. Available: {", ".join(TOOLS)}'
    try:
        return TOOLS[name](**args)
    except Exception as e:
        return f'Error: {type(e).__name__}: {e}'

# %%
# The model's side of tool calling. Qwen3 was trained to answer a tool request in one fixed format:
#   <tool_call>
#   {"name": "read_file", "arguments": {"path": "recall/summary.csv"}}
#   </tool_call>
# The chat template writes the tool schemas into the prompt; parse_tool_calls reads that format back out.
TOOL_CALL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)

def parse_tool_calls(text):
    calls = []
    for m in TOOL_CALL.finditer(text):
        try:
            c = json.loads(m.group(1))
            calls.append(dict(name=c['name'], arguments=c.get('arguments', {})))
        except (json.JSONDecodeError, KeyError) as e:
            calls.append(dict(name='invalid', arguments={}, error=f'could not parse {m.group(1)!r}: {e}'))
    return calls

@torch.no_grad()
def generate(messages):
    prompt = tok.apply_chat_template(messages, tools=list(TOOLS.values()), add_generation_prompt=True,
                                     tokenize=False, enable_thinking=THINKING)
    ids = tok(prompt, return_tensors='pt').to(model.device)
    out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    reply = tok.decode(out[0, ids['input_ids'].shape[1]:], skip_special_tokens=True)
    return reply, ids['input_ids'].shape[1]

# %%
# The loop. This is the whole agent.
SYSTEM = ('You are a research assistant for a lab that trains small language models and benchmarks them. '
          'Answer from the files, not from memory: list folders to find files, read them, then answer. '
          'Quote the numbers you use and name the file they came from.')

def run_agent(task, verbose=True):
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=task)]
    for step in range(1, MAX_STEPS + 1):
        reply, ctx = generate(messages)
        calls = parse_tool_calls(reply)
        if verbose:
            print(f'\n── step {step} · context {ctx:,} tokens ──\n{reply.strip()}')
        if not calls:                                         # no tool requested: this is the final answer
            return reply.strip()
        messages.append(dict(role='assistant', content=reply,
                             tool_calls=[dict(type='function', function=dict(name=c['name'], arguments=c['arguments']))
                                         for c in calls]))
        for c in calls:
            result = c['error'] if 'error' in c else run_tool(c['name'], c['arguments'])
            if verbose:
                print(f'  → {c["name"]}({json.dumps(c["arguments"])}) returned {len(result):,} characters')
            messages.append(dict(role='tool', name=c['name'], content=result))
    return f'Stopped after {MAX_STEPS} steps without a final answer.'

# %%
# Try it. Change the task to anything about the files in ROOT.
answer = run_agent('In the recall test results, which model copies best when the first copy is far back, '
                   'and how does attn-gqa compare with llama-style?')
print('\n══ answer ══\n' + answer)

# %% [markdown]
# ## What to look for in the trace
# - **The first step is a tool call, not an answer.** The system prompt and the tool schemas steered it to look before
#   answering. Remove the tools (`TOOLS = {}`) and it will answer from memory: fluent and wrong.
# - **Context grows every step.** Each tool result is appended and stays. At 4,000 characters per read that's about
#   1,000 tokens per file, so a 20-step task can reach 16K+ tokens: the range the Qwen3 recall test checks.
# - **Errors are fed back, not raised.** Ask it to read a file that doesn't exist and watch it recover by listing the
#   folder first.
# - **Where a 1.7B model breaks:** malformed tool calls, re-reading the same file, or misreading a table. Those failures
#   are what the next step (fine-tuning on tool-use examples) is for, and this trace is how you will measure the fix.
