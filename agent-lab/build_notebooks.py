"""Build the Colab notebooks from the percent-format scripts, so the notebook and the script never drift apart.

    python build_notebooks.py

writes  qwen3_recall.ipynb    long-context recall of the agent's model, at bf16 / 8-bit / 4-bit
        mini_agent.ipynb      a model, two tools and a loop
        agent_finetune.ipynb  QLoRA fine-tuning on tool-use traces, scored before and after

A cell containing `from agent_core import *` is replaced by the cells of agent_core.py, so both agent notebooks give
the model exactly the same tools, schemas and loop.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

def parse(text):
    """Percent-format cells → list of (kind, source)."""
    cells = []
    for chunk in re.split(r'^# %%', text, flags=re.M)[1:]:
        head, _, body = chunk.partition('\n')
        cells.append(('markdown' if head.strip() == '[markdown]' else 'code', body.strip('\n')))
    return cells

def read(name):
    return open(os.path.join(HERE, name), encoding='utf-8').read()

def expand(cells):
    core = parse(read('agent_core.py'))
    out = []
    for kind, body in cells:
        out += core if kind == 'code' and body.startswith('from agent_core import *') else [(kind, body)]
    return out

def write_notebook(path, cells, gpu):
    nb_cells = []
    for kind, body in cells:
        lines = body.split('\n')
        if kind == 'markdown':
            lines = [l[2:] if l.startswith('# ') else l.lstrip('#') for l in lines]
        src = [l + '\n' for l in lines[:-1]] + lines[-1:]
        cell = dict(cell_type=kind, metadata={}, source=src)
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    nb = dict(nbformat=4, nbformat_minor=5, cells=nb_cells, metadata=dict(
        accelerator='GPU', colab=dict(provenance=[], gpuType=gpu),
        kernelspec=dict(name='python3', display_name='Python 3'), language_info=dict(name='python')))
    json.dump(nb, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

for name, gpu in [('qwen3_recall', 'L4'), ('mini_agent', 'T4'), ('agent_finetune', 'L4')]:
    cells = expand(parse(read(f'{name}.py')))
    assert not any('from agent_core' in body for kind, body in cells if kind == 'code'), name
    write_notebook(os.path.join(HERE, f'{name}.ipynb'), cells, gpu)
    print(f'{name}.ipynb: {len(cells)} cells')
