"""Build the Colab notebooks from the percent-format scripts, so the notebook and the script never drift apart.

    python build_notebooks.py

writes  qwen3_recall.ipynb   long-context recall of the agent's model, at bf16 / 8-bit / 4-bit
        mini_agent.ipynb     a model, two tools and a loop
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

for name, gpu in [('qwen3_recall', 'L4'), ('mini_agent', 'T4')]:
    cells = parse(open(os.path.join(HERE, f'{name}.py'), encoding='utf-8').read())
    write_notebook(os.path.join(HERE, f'{name}.ipynb'), cells, gpu)
    print(f'{name}.ipynb: {len(cells)} cells')
