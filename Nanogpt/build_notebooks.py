"""
build_notebooks.py — generate one runnable Jupyter notebook per nano script.

Each notebook is derived from its .py twin so they never drift apart:
  - the module docstring becomes the intro markdown cell
  - every "SECTION" comment banner becomes a markdown cell
  - the code between banners becomes code cells
  - argparse is replaced by a QUICK flag cell (notebooks have no CLI)
  - plots are displayed inline at the end

Re-run this file whenever a script changes:  python build_notebooks.py
"""
import re, os, nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = re.compile(r"^# -{40,}\s*$")

def md(src):  return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

def convert(pyfile, title, quick_note, tail_cells=()):
    src = open(os.path.join(HERE, pyfile), encoding="utf-8").read()

    # 1) pull the module docstring -> intro markdown (fenced to keep layout)
    m = re.match(r'\s*"""(.*?)"""\s*\n', src, re.S)
    docstring, rest = (m.group(1), src[m.end():]) if m else ("", src)

    # 2) notebook-only patches
    rest = re.sub(r"(?m)^p = argparse\.ArgumentParser\(\)\n(?:^p\.add_argument.*\n)+^args = p\.parse_args\(\)\n", "", rest)
    rest = rest.replace("import argparse, ", "import ")
    rest = rest.replace("args.quick", "QUICK")
    rest = re.sub(r"(?m)^HERE = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)$",
                  "HERE = os.getcwd()          # notebook runs from the Nanogpt folder", rest)
    rest = re.sub(r"(?m)^\s*matplotlib\.use\(\"Agg\"\)\n", "", rest)

    # 3) split on section banners:  dashline / "# ..." lines / dashline
    lines = rest.split("\n")
    cells, buf, i = [], [], 0
    def flush():
        chunk = "\n".join(buf).strip("\n")
        if chunk.strip(): cells.append(code(chunk))
        buf.clear()
    while i < len(lines):
        if DASH.match(lines[i]) and i + 1 < len(lines) and lines[i+1].startswith("# "):
            flush()
            i += 1
            banner = []
            while i < len(lines) and lines[i].startswith("#") and not DASH.match(lines[i]):
                banner.append(lines[i].lstrip("#").strip()); i += 1
            if i < len(lines) and DASH.match(lines[i]): i += 1
            head, body = banner[0], banner[1:]
            text = f"### {head}\n" + "\n".join(body)
            cells.append(md(text))
        else:
            buf.append(lines[i]); i += 1
    flush()

    # 4) assemble
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        md(f"# {title}\n\n```text\n{docstring.strip()}\n```"),
        md("## ⚙ Run mode\n`QUICK = True` gives a few-minute demo run so you can watch the "
           "mechanics work end to end. Set `QUICK = False` for the full experiment"
           + (f" — {quick_note}" if quick_note else "") + "."),
        code("QUICK = True   # ← flip to False for the full run"),
        *cells,
        *tail_cells,
    ]
    out = os.path.join(HERE, pyfile.replace(".py", "").replace("nano", "nano_").replace("nano__", "nano_") + ".ipynb")
    return nb, out

JOBS = [
    ("nanogpt_attention.py", "nanoGPT — causal self-attention, end to end",
     "expect recognisable Shakespeare after the full 3000 steps", ()),
    ("nanobert.py", "nanoBERT — delete the mask, hide the words",
     "the fill-in-the-blank demo gets much better with the full run", ()),
    ("nano_scaling_laws.py", "nano Scaling Laws — the Kaplan ladder",
     "5 model sizes instead of 4, trained 4x longer",
     (md("## The picture"),
      code("from IPython.display import Image, display\n"
           "display(Image('runs/scaling_laws.png'))"))),
    ("nano_chinchilla.py", "nano Chinchilla — IsoFLOP valleys",
     "3 budgets × 6 model sizes for a much better exponent fit",
     (md("## The picture"),
      code("from IPython.display import Image, display\n"
           "display(Image('runs/chinchilla_isoflop.png'))"))),
    ("nano_rope.py", "nano RoPE — rotate, don't memorise",
     "deeper/wider models make the extrapolation gap even clearer",
     (md("## The picture"),
      code("from IPython.display import Image, display\n"
           "display(Image('runs/rope_extrapolation.png'))"))),
    ("nano_llama.py", "nano LLaMA — the modern recipe vs GPT-2",
     "the architectural gap grows with training length",
     (md("## The picture"),
      code("from IPython.display import Image, display\n"
           "display(Image('runs/llama_vs_gpt.png'))"))),
]

for pyfile, title, note, tail in JOBS:
    nb, out = convert(pyfile, title, note, tail)
    nbf.write(nb, out)
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    n_co = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"{os.path.basename(out):38s} {n_md:2d} md + {n_co:2d} code cells")

# compare.py -> notebook (no QUICK cell needed)
src = open(os.path.join(HERE, "compare.py"), encoding="utf-8").read()
m = re.match(r'\s*"""(.*?)"""\s*\n', src, re.S)
docstring, rest = m.group(1), src[m.end():]
rest = re.sub(r"(?m)^HERE = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)$",
              "HERE = os.getcwd()", rest)
nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.cells = [nbf.v4.new_markdown_cell(f"# Compare — claim-by-claim verdicts\n\n```text\n{docstring.strip()}\n```\n\n"
            "Run the four notebooks (or scripts) first; this reads `runs/*.json`."),
            nbf.v4.new_code_cell(rest.strip())]
nbf.write(nb, os.path.join(HERE, "compare.ipynb"))
print("compare.ipynb                          1 md +  1 code cells")
print("done")
