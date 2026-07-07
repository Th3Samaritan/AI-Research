# Build the BERT math PDF: one matplotlib figure + print HTML for headless Chrome.
import base64, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK="#0A0D13"; PANEL="#141A28"; LINE="#262E43"; TXT="#E8ECF5"; MUT="#8B95AC"
AMBER="#F5A524"; GOOD="#5BE3A6"; BAD="#EF5B7B"
plt.rcParams.update({"figure.facecolor":INK,"axes.facecolor":INK,"savefig.facecolor":INK,
  "axes.edgecolor":LINE,"axes.labelcolor":MUT,"text.color":TXT,"xtick.color":MUT,
  "ytick.color":MUT,"font.size":11,"figure.dpi":200})

# --- figure: the two attention masks ---
T=12; rng=np.random.default_rng(7)
raw=np.exp(rng.normal(0,1,(T,T)))
causal=np.tril(np.ones((T,T)))
def norm(m):
    m=m/m.sum(1,keepdims=True); return m
fig,axes=plt.subplots(1,2,figsize=(6.8,3.1))
for ax,mask,name,cmap in [(axes[0],causal,"GPT — causal mask","Reds"),
                          (axes[1],np.ones((T,T)),"BERT — no mask","Greens")]:
    ax.imshow(norm(raw*mask),cmap=cmap,vmin=0)
    ax.set_title(name,color=TXT,fontsize=10)
    ax.set_xlabel("attends to position j"); ax.set_ylabel("token position t")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(LINE)
fig.tight_layout(); fig.savefig("fig_masks.png",bbox_inches="tight",pad_inches=0.12)
plt.close(fig); print("wrote fig_masks.png")

def b64(p):
    with open(p,"rb") as f: return "data:image/png;base64,"+base64.b64encode(f.read()).decode()
FIGMASK=b64("fig_masks.png")

SECTIONS=[
 ("The one-line difference","§3 · Architecture",[
  ("att","self-attention, shared by GPT and BERT",
   r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}+M\right)V",
   "Every token forms a query Q, a key K and a value V; the scaled dot-products decide how much each position reads from every other. The entire GPT/BERT architectural split is the mask M: GPT sets M = −∞ above the diagonal so no token sees its future; BERT sets M = 0 — the mask is deleted and every token attends in both directions. That deletion is why BERT represents language better, and why it cannot generate text: it was never trained to predict without seeing the right-hand context.",
   [("Q,K,V","learned linear projections of the input","—"),
    ("√d_k","scaling that keeps softmax gradients healthy","—"),
    ("M (GPT)","−∞ above the diagonal — causal","autoregressive"),
    ("M (BERT)","all zeros — bidirectional","understanding")]),
 ],True,"The same random attention scores under the two masks: GPT's world is triangular; BERT's is the full square."),

 ("Masked language modelling","§3.1 · Task #1",[
  ("MLM","the objective",
   r"\mathcal{L}_{\mathrm{MLM}}=-\sum_{i\in\mathcal{M}}\log P\big(x_i\mid\tilde{x}\big),\qquad |\mathcal{M}|\approx 0.15\,T",
   "Bidirectional attention makes next-word prediction cheatable (each position could look the answer up), so BERT invents a new exam: corrupt the input x into x̃, then reconstruct the ORIGINAL tokens at the corrupted 15% of positions only. The loss is ordinary cross-entropy, restricted to the set M of selected positions.",
   [("M","the ~15% selected positions","15%"),
    ("x̃","the corrupted input sequence","—")]),
  ("80/10/10","the corruption recipe",
   r"\text{for } i\in\mathcal{M}:\quad 80\%\to[\mathrm{MASK}],\quad 10\%\to\text{random token},\quad 10\%\to\text{unchanged}",
   "Why not replace all selected tokens with [MASK]? Because [MASK] never appears at fine-tuning or inference time — a model trained 100% on it would learn that only masked slots carry gradient. The 10% random substitutions and 10% left-unchanged (but still graded) force the encoder to maintain an honest contextual representation of every position, since it never knows which ones will be examined.",
   [("80%","replaced by [MASK]","—"),
    ("10%","replaced by a random vocabulary token","—"),
    ("10%","left as-is, still predicted","—")]),
 ],False,None),

 ("NSP and the input recipe","§3.1 · Task #2",[
  ("NSP","next sentence prediction",
   r"\mathcal{L}_{\mathrm{NSP}}=-\log P\big(\mathrm{IsNext}\mid \mathbf{C}\big),\qquad \mathbf{C}=h_{[\mathrm{CLS}]}",
   "50% of training pairs are consecutive sentences, 50% random; a binary classifier on the final [CLS] vector must tell them apart — intended to teach sentence-level relations for QA and NLI. History's verdict: the paper's own ablations show modest gains and RoBERTa (2019) dropped NSP entirely without loss. The MLM objective is the part of BERT that mattered.",
   [("C","final hidden state of [CLS]","sequence summary")]),
  ("input","the input representation",
   r"\mathrm{input}_i=E_{\mathrm{tok}}(x_i)+E_{\mathrm{seg}}(A/B)+E_{\mathrm{pos}}(i)",
   "Three embeddings summed per token: WordPiece identity (30,000-piece subword vocabulary), a segment flag marking sentence A or B, and a learned position embedding. Special tokens [CLS] (front, sequence-level summary) and [SEP] (sentence boundaries) give every downstream task the same plug-in shape.",
   [("WordPiece","subword vocabulary","30k"),
    ("[CLS]/[SEP]","summary slot / boundary marker","—")]),
 ],False,None),

 ("Architecture & pre-training","§3 · The numbers",[
  ("size","parameter budget",
   r"N_{\text{non-emb}}\approx 12\,L\,H^{2}\qquad\text{Base: }12{\cdot}12{\cdot}768^2\!\approx\!85\mathrm{M}\;(+\,\text{emb}\approx 110\mathrm{M})",
   "BERT-Base: L=12 layers, H=768, A=12 heads, feed-forward 4H — 110M parameters, deliberately sized to match GPT-1 for a fair comparison. BERT-Large: L=24, H=1024, A=16 — 340M, the configuration behind the headline results. Pre-trained on BooksCorpus + English Wikipedia (3.3B words) for 40 epochs with GELU activations and Adam. In post-Chinchilla hindsight (~30 tokens/param), Base was accidentally near compute-optimal.",
   [("Base","L12 · H768 · A12","110M params"),
    ("Large","L24 · H1024 · A16","340M params"),
    ("data","BooksCorpus + Wikipedia","3.3B words")]),
 ],False,None),

 ("Fine-tuning heads","§4 · Transfer",[
  ("cls","classification head (GLUE)",
   r"P(y\mid\text{sent})=\mathrm{softmax}(W\,\mathbf{C}),\qquad W\in\mathbb{R}^{K\times H}",
   "Sentiment, entailment, similarity: one new K×H matrix on the [CLS] vector, then fine-tune everything for 2–4 epochs. A few thousand new parameters steering 110M pre-trained ones. GLUE at publication: 80.5%, +7.7 points over the prior state of the art.",
   [("W","the only new parameters","K·H")]),
  ("span","span head (SQuAD)",
   r"P_{\mathrm{start}}(i)=\frac{e^{S\cdot h_i}}{\sum_j e^{S\cdot h_j}},\quad P_{\mathrm{end}}(i)=\frac{e^{E\cdot h_i}}{\sum_j e^{E\cdot h_j}},\quad (\hat{\jmath},\hat{k})=\arg\max_{j\le k}S{\cdot}h_j+E{\cdot}h_k",
   "Extractive QA: the answer is a span of the passage. Two learned vectors S and E score every token as a candidate start/end; the best valid pair (start before end) wins. Only 2·H new parameters — 93.2 F1 on SQuAD v1.1, surpassing human performance. The pattern across all heads: bidirectional pre-training did the hard work; the task head is nearly free. The key ablation: the same model with a left-to-right mask drops from 84.9 to 77.8 F1 — bidirectionality is the ingredient.",
   [("S, E","start / end scoring vectors","2·H params"),
    ("ablation","LTR (causal) vs bidirectional on SQuAD","77.8 vs 84.9 F1")]),
 ],False,None),
]

def params_html(ps):
    rows="".join(f'<tr><td class="sym">$${s}$$</td><td>{d}</td><td class="val">{v}</td></tr>' for s,d,v in ps)
    return f'<table class="params"><thead><tr><th>symbol</th><th>meaning</th><th>value</th></tr></thead><tbody>{rows}</tbody></table>'

def section_html(title,kicker,eqs,fig,cap):
    cards=""
    for num,name,latex,does,ps in eqs:
        cards+=f'''<div class="eqcard"><div class="eqhead"><span class="eqname">{name}</span><span class="eqno">({num})</span></div>
        <div class="eq">$${latex}$$</div><div class="does"><b>What it does.</b> {does}</div>{params_html(ps)}</div>'''
    figblock=f'<figure><img src="{FIGMASK}"><figcaption>{cap}</figcaption></figure>' if fig else ""
    return f'<section><div class="skicker">{kicker}</div><h2>{title}</h2>{cards}{figblock}</section>'

body="".join(section_html(*s) for s in SECTIONS)

HTML=f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>BERT — the math explained</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<style>
  @page {{ size: A4; margin: 15mm 14mm 16mm; }}
  :root{{--ink:#0A0D13;--panel:#141A28;--panel2:#19202F;--line:#262E43;--txt:#E8ECF5;
    --mut:#8B95AC;--mut2:#5C6680;--amber:#F5A524;--cool:#5B8DEF;--good:#5BE3A6;}}
  *{{box-sizing:border-box}}
  html,body{{margin:0;padding:0;background:var(--ink);color:var(--txt);
    font-family:"Segoe UI",system-ui,sans-serif;-webkit-print-color-adjust:exact;print-color-adjust:exact}}
  .katex{{color:var(--txt)}}
  .cover{{height:250mm;display:flex;flex-direction:column;justify-content:center;
    background:radial-gradient(600px 300px at 80% 10%,#17223c 0%,transparent 60%),
    radial-gradient(500px 260px at 0% 100%,#1a1430 0%,transparent 55%),var(--ink);
    border:1px solid var(--line);border-radius:14px;padding:30mm 22mm;page-break-after:always}}
  .cover .tag{{font-family:"Consolas",monospace;font-size:11px;letter-spacing:.3em;color:var(--amber);text-transform:uppercase}}
  .cover h1{{font-size:46px;line-height:1.05;letter-spacing:-.02em;margin:14px 0 8px;font-weight:800}}
  .cover h1 .g{{background:linear-gradient(92deg,var(--amber),#FFC85C 45%,var(--cool));
    -webkit-background-clip:text;background-clip:text;color:transparent}}
  .cover .sub{{font-size:16px;color:var(--mut);line-height:1.6;max-width:150mm}}
  .cover .meta{{margin-top:26px;font-family:"Consolas",monospace;font-size:12px;color:var(--mut2);line-height:1.9}}
  .cover .meta b{{color:var(--txt)}}
  .cover .rule{{height:2px;width:70mm;background:linear-gradient(90deg,var(--amber),transparent);margin:18px 0}}
  section{{page-break-inside:avoid;margin:0 0 14px}}
  .skicker{{font-family:"Consolas",monospace;font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:var(--amber);margin-top:6px}}
  h2{{font-size:22px;letter-spacing:-.01em;margin:2px 0 12px;font-weight:700;border-bottom:1px solid var(--line);padding-bottom:7px}}
  .eqcard{{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);
    border-left:3px solid var(--amber);border-radius:11px;padding:12px 15px;margin:0 0 11px;page-break-inside:avoid}}
  .eqhead{{display:flex;justify-content:space-between;align-items:baseline}}
  .eqname{{font-family:"Consolas",monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--amber)}}
  .eqno{{font-family:"Consolas",monospace;font-size:11px;color:var(--mut2)}}
  .eq{{text-align:center;padding:8px 4px 10px;font-size:16px}}
  .does{{font-size:12px;line-height:1.6;color:var(--mut);margin-bottom:9px}}
  .does b{{color:var(--txt)}}
  table.params{{width:100%;border-collapse:collapse;font-size:11px}}
  table.params th{{text-align:left;font-family:"Consolas",monospace;font-size:8.5px;letter-spacing:.1em;
    text-transform:uppercase;color:var(--mut2);border-bottom:1px solid var(--line);padding:3px 8px}}
  table.params td{{padding:4px 8px;border-bottom:1px solid #1C2333;color:var(--mut);vertical-align:top}}
  table.params td.sym{{color:var(--amber);width:80px}}
  table.params td.val{{color:var(--good);font-family:"Consolas",monospace;text-align:right;white-space:nowrap;width:120px}}
  figure{{margin:8px 0 4px;text-align:center;page-break-inside:avoid}}
  figure img{{width:100%;max-width:160mm;border:1px solid var(--line);border-radius:10px}}
  figcaption{{font-family:"Consolas",monospace;font-size:9.5px;color:var(--mut2);margin-top:5px}}
  .foot{{margin-top:10px;border-top:1px solid var(--line);padding-top:8px;
    font-family:"Consolas",monospace;font-size:9px;color:var(--mut2);line-height:1.7}}
</style></head>
<body>
  <div class="cover">
    <div class="tag">Paper → Code · arXiv:1810.04805</div>
    <h1>BERT<br><span class="g">Bidirectional Transformers</span></h1>
    <div class="rule"></div>
    <div class="sub">Every equation of Devlin, Chang, Lee &amp; Toutanova (Google AI, 2018), explained
      parameter by parameter — the deleted causal mask, the 15% / 80-10-10 masked-language-model
      objective, NSP, the input recipe, and the fine-tuning heads that set 11 states of the art.</div>
    <div class="meta">
      <b>The math explained</b> — companion to the interactive site<br>
      Original authors: J. Devlin, M.-W. Chang, K. Lee, K. Toutanova<br>
      Visualised &amp; annotated by <b>Abdulsamad Teniola Muyideen</b><br>
      7 core equations · Base/Large configs · runnable char-level twin at Nanogpt/nanobert.py
    </div>
  </div>
  {body}
  <div class="foot">
    BERT — the math explained · arXiv:1810.04805 · © Abdulsamad Teniola Muyideen ·
    sister documents: "Attention, Animated" (Vaswani 2017), "Scaling Laws" (Kaplan 2020), "Chinchilla" (Hoffmann 2022).
  </div>
  <script>
    document.addEventListener("DOMContentLoaded",function(){{
      renderMathInElement(document.body,{{delimiters:[{{left:"$$",right:"$$",display:false}}],throwOnError:false}});
      document.title="rendered";
    }});
  </script>
</body></html>'''

open("_pdf.html","w",encoding="utf-8").write(HTML)
print("wrote _pdf.html",len(HTML),"bytes")
