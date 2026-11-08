# %% [markdown]
# ## The agent core: tools, tool-call parsing and the loop
# Shared by `mini_agent` and `agent_finetune` (build_notebooks.py pastes this file into both), so the fine-tuned model
# is trained on exactly the prompt it will see when used.

# %%
# Tools. Plain Python functions. The type hints and the docstring are not decoration: transformers turns them into the
# JSON schema that goes into the prompt, and that schema is all the model knows about each tool.
import os, re, json

ROOT = os.path.realpath('.')
READ_LIMIT = 4000        # characters a single read_file returns: every result stays in the context for good

def set_root(path):
    """Point the tools at a folder. They cannot see anything outside it."""
    global ROOT
    ROOT = os.path.realpath(path)

def safe_path(path: str) -> str:
    """Keep every tool inside ROOT, so the model can't read anything else on the machine."""
    full = os.path.realpath(os.path.join(ROOT, path))
    if os.path.commonpath([full, ROOT]) != ROOT:
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

def sort_table(path: str, column: str, descending: bool = True) -> str:
    """
    Sort a CSV table by one numeric column and list every run with its value, in order. Use it to find the highest or
    lowest value instead of comparing numbers by eye.

    Args:
        path: Path to a .csv file relative to the lab folder, for example "results/summary.csv".
        column: The column to sort by, written exactly as in the table's header.
        descending: true lists the highest value first, false the lowest first.
    """
    import csv
    if isinstance(descending, str):
        descending = descending.strip().lower() not in ('false', '0', 'no')
    with open(safe_path(path), encoding='utf-8', errors='replace', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return f'Error: {path} has no rows'
    header = list(rows[0])
    if column not in header:
        return f'Error: there is no column {column!r} in {path}. Columns: {", ".join(header)}'
    key = next((h for h in ('run', 'name', 'model') if h in header), header[0])
    def value(r):
        try:
            return float(r[column])
        except (TypeError, ValueError):
            return None
    have = sorted((r for r in rows if value(r) is not None), key=value, reverse=descending)
    blank = len(rows) - len(have)
    return (f'{path} sorted by {column}, {"highest" if descending else "lowest"} first '
            f'({len(have)} rows{f", {blank} blank" if blank else ""}):\n{key},{column}\n'
            + '\n'.join(f'{r[key]},{r[column]}' for r in have))

TOOLS = {f.__name__: f for f in (list_files, read_file, sort_table)}

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
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            calls.append(dict(name='invalid', arguments={}, error=f'Error: could not parse {m.group(1)!r}: {e}'))
    return calls

def assistant_turn(reply, calls):
    """The message stored for a reply that called tools. The chat template writes each call from `tool_calls`, so the
    <tool_call> text comes out of `content`: left in, every call would appear twice in the history."""
    return dict(role='assistant', content=TOOL_CALL.sub('', reply).strip(),
                tool_calls=[dict(type='function', function=dict(name=c['name'], arguments=c['arguments'])) for c in calls])

# %%
# The loop. This is the whole agent. `generate(messages)` returns (reply, prompt length in tokens).
SYSTEM = ('You are a research assistant for a lab that trains small language models and benchmarks them. '
          'Answer from the files, not from memory: list folders to find files, read them, then answer. '
          'For the highest or lowest value in a table, use sort_table. '
          'Quote the numbers you use and name the file they came from.')

def run_agent(task, generate, max_steps=8, verbose=True):
    messages = [dict(role='system', content=SYSTEM), dict(role='user', content=task)]
    n_calls = bad_calls = 0
    for step in range(1, max_steps + 1):
        reply, ctx = generate(messages)
        calls = parse_tool_calls(reply)
        if verbose:
            print(f'\n── step {step} · context {ctx:,} tokens ──\n{reply.strip()}')
        if not calls:                                         # no tool requested: this is the final answer
            return dict(answer=reply.strip(), steps=step, calls=n_calls, bad_calls=bad_calls, context=ctx, messages=messages)
        messages.append(assistant_turn(reply, calls))
        for c in calls:
            result = c['error'] if 'error' in c else run_tool(c['name'], c['arguments'])
            n_calls += 1
            bad_calls += result.startswith('Error')
            if verbose:
                print(f'  → {c["name"]}({json.dumps(c["arguments"])}) returned {len(result):,} characters')
            messages.append(dict(role='tool', name=c['name'], content=result))
    return dict(answer=None, steps=max_steps, calls=n_calls, bad_calls=bad_calls, context=ctx, messages=messages)
