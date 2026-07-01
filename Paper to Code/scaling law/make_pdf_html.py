# Assemble the print-quality HTML for the Scaling Laws PDF (converted via headless Chrome).
import base64, os

def b64(p):
    with open(p,"rb") as f: return "data:image/png;base64,"+base64.b64encode(f.read()).decode()

FIG = {k:b64(f"fig_{k}.png") for k in ["basics","batch","nd","train","alloc","breakdown"]}

# ---- equation blocks: (number, name, latex, "what it does", [ (sym, meaning, value) ... ]) ----
SECTIONS = [
 ("The three basic power laws", "§3 · Empirical Results", [
  ("1.1","L(N) — parameter law",
   r"L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N}",
   "Predicts the converged test loss of a model with N non-embedding parameters when the dataset is large enough that data is not the bottleneck. On a log-log plot this is a straight line: multiplying N by any factor drops the loss by a fixed factor. Doubling N multiplies the loss by 2^-0.076 ≈ 0.95 (a 5% reduction).",
   [("N","number of non-embedding parameters (excludes token & positional embeddings)","—"),
    ("α_N","parameter power-law exponent","0.076"),
    ("N_c","parameter scale — tokenizer-dependent, no fundamental meaning","8.8×10¹³")]),
  ("1.2","L(D) — data law",
   r"L(D)=\left(\frac{D_c}{D}\right)^{\alpha_D}",
   "The loss of a large model trained with early stopping on a limited dataset of D tokens — now data is the bottleneck. Same functional shape as L(N) but a steeper slope, so each doubling of data buys slightly more than each doubling of parameters.",
   [("D","dataset size in tokens","—"),
    ("α_D","data power-law exponent","0.095"),
    ("D_c","data scale (tokens)","5.4×10¹³")]),
  ("1.3","L(C_min) — compute law",
   r"L(C_{\min})=\left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}}",
   "The best loss reachable for a given budget of optimally-allocated compute: a sufficiently large dataset, an optimally-sized model, and a batch size well below B_crit. This is the trend to use for real compute forecasts (the naive fixed-batch trend L(C) is messier).",
   [("C_min","optimal non-embedding compute (PF-days; 1 PF-day = 8.64×10¹⁹ FLOP)","—"),
    ("α_C^min","optimal-compute exponent","0.050"),
    ("C_c^min","compute scale (PF-days)","3.1×10⁸")]),
  ("3.3","L(C) — naive fixed-batch compute law",
   r"L(C)=\left(\frac{C_c}{C}\right)^{\alpha_C}",
   "The empirical loss versus raw training compute C when the batch size is held fixed. Slightly steeper and lumpier than the L(C_min) trend; retained only for comparison.",
   [("C","raw training compute (PF-days), C = 6·N·B·S","—"),
    ("α_C","naive compute exponent","0.057"),
    ("C_c","naive compute scale (PF-days)","1.6×10⁷")]),
 ], "basics", "L(N) amber · L(D) blue · L(C_min) green — each a straight line over 6–8 orders of magnitude."),

 ("Compute & parameter counting", "§2 · Background", [
  ("2.1","non-embedding parameter count",
   r"N \approx 2\,d_{model}\,n_{layer}\,(2\,d_{attn}+d_{ff}) = 12\,n_{layer}\,d_{model}^{2}",
   "How the model size N is computed from the architecture, using the standard proportions d_attn = d_ff/4 = d_model. Embedding and positional parameters are deliberately excluded — doing so is what makes the scaling laws clean.",
   [("n_layer","number of Transformer layers","—"),
    ("d_model","residual-stream width","—"),
    ("d_ff, d_attn","feed-forward and attention widths","d_ff=4·d_model")]),
  ("2.2","compute per token",
   r"C_{forward}\approx 2N + 2\,n_{layer}\,n_{ctx}\,d_{model}\;;\qquad C \approx 6N \ \text{per token}",
   "Forward-pass FLOP per token; the factor of 2 is the multiply-accumulate. Adding the backward pass (≈2× forward) gives the rule of thumb C ≈ 6N FLOP per token, which underlies C = 6·N·B·S for a whole run.",
   [("n_ctx","context length in tokens","1024"),
    ("B","batch size (tokens)","—"),
    ("S","number of parameter updates (steps)","—")]),
 ], None, None),

 ("Critical batch size", "§5.1 · Batch adjustment", [
  ("1.4","B_crit(L) — critical batch size",
   r"B_{\mathrm{crit}}(L)=\frac{B_*}{L^{1/\alpha_B}}",
   "The batch size giving a roughly optimal time/compute trade-off. Batches up to B_crit cost almost no extra compute; beyond it you get diminishing returns. It depends only on the current loss L — not on model size — and roughly doubles for every 13% drop in loss, so as training improves you can parallelise harder.",
   [("L","current loss (nats/token)","—"),
    ("B_*","batch scale","2.1×10⁸ tok"),
    ("α_B","batch exponent (1/α_B ≈ 4.8)","0.21")]),
  ("5.4","standardised steps S_min",
   r"S_{\min}=\frac{S}{1+B_{\mathrm{crit}}(L)/B}",
   "Converts the actual number of steps S taken at batch size B into the minimum steps needed at B ≫ B_crit. This universal S_min is what the learning-curve fits are parameterised in.",
   [("S","steps actually taken at batch B","—"),
    ("B","the batch size used","—")]),
  ("5.5","standardised compute C_min",
   r"C_{\min}=\frac{C}{1+B/B_{\mathrm{crit}}(L)}",
   "Converts raw compute C = 6·N·B·S into the minimum compute you would have spent at B ≪ B_crit — the clean quantity used for all forecasts.",
   [("C","raw compute spent (PF-days)","—")]),
 ], "batch", "B_crit shoots up as the loss falls, diverging as L→0 near the entropy floor."),

 ("Model size × data — overfitting", "§4 · Infinite-Data Limit", [
  ("1.5","L(N,D) — the unified law",
   r"L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}",
   "A single equation for both knobs at once. As D→∞ it collapses to L(N); as N→∞ it collapses to L(D). In between, whichever resource is scarcer dominates. Built so a change of vocabulary merely rescales N_c and D_c.",
   [("N","params","—"),("D","tokens","—"),
    ("α_N/α_D","coupling ratio","≈ 0.80")]),
  ("4.3","δL — the overfitting penalty",
   r"\delta L(N,D)=\frac{L(N,D)}{L(N,\infty)}-1=\left[1+\left(\frac{N}{N_c}\right)^{\alpha_N/\alpha_D}\frac{D_c}{D}\right]^{\alpha_D}-1",
   "The fractional extra loss caused by a finite dataset, relative to the infinite-data floor L(N). Remarkably it depends only on the single combination ~ N^0.74/D, so every model size collapses onto one universal curve.",
   [("L(N,∞)","= L(N), the infinite-data floor","—")]),
  ("4.4","no-overfit data requirement",
   r"D \;\gtrsim\; (5\times10^{3})\,N^{0.74}",
   "The minimum dataset size that keeps overfitting under the ~0.02 random-seed noise floor. Data need only grow sub-linearly in model size: grow N by 8× and D by just ~5×. The exponent 0.74 = 1 − α_N/α_D.",
   [("0.74","= 1 − α_N/α_D","—")]),
 ], "nd", "Solid = finite-data L(N,D); dotted = capacity floor L(N). Where they meet, more data stops helping."),

 ("Model size × training time", "§5 · Training Time", [
  ("1.6","L(N,S_min) — learning-curve law",
   r"L(N,S_{\min})=\left(\frac{N_c}{N}\right)^{\alpha_N}+\left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}",
   "Loss as a capacity floor (first term, set by N) plus a training-progress term (second term, falling with steps). Because the shape is nearly independent of N, you can fit the early part of a run and extrapolate where it will end up.",
   [("S_min","steps measured at B ≫ B_crit","—"),
    ("α_S","step exponent","0.76"),
    ("S_c","step scale","2.1×10³")]),
  ("5.7","early-stopping lower bound",
   r"S_{\mathrm{stop}}(N,D)\;\gtrsim\;\frac{S_c}{\big[L(N,D)-L(N,\infty)\big]^{1/\alpha_S}}",
   "A lower-bound estimate of the step at which to stop when training is data-limited: the larger the finite-data gap over the infinite-data floor, the sooner you should stop.",
   [("L(N,D)−L(N,∞)","the overfitting gap","—")]),
 ], "train", "Bigger models (brighter) start and finish lower, and reach any target loss in fewer steps."),

 ("Optimal allocation of compute", "§6 · Compute Budget", [
  ("1.7","compute-efficient frontier",
   r"N_{\mathrm{opt}}\propto C_{\min}^{0.73},\quad B\propto C_{\min}^{0.24},\quad S_{\min}\propto C_{\min}^{0.03},\quad D_{\mathrm{opt}}\propto C_{\min}^{0.27}",
   "How to spend a budget. Model size rockets up (exponent 0.73) while serial steps barely move (0.03). A billion-fold more compute should become an ~billion-fold bigger model with almost no extra training time — the paper's central practical message.",
   [("N_e","model scale","1.3×10⁹"),("B_e","batch scale","2.0×10⁶"),
    ("S_e","step scale","5.4×10³"),("D_e","data scale","2.0×10¹⁰")]),
  ("6.4","the exponent, predicted",
   r"\alpha_C^{\min}=\frac{1}{\,1/\alpha_S+1/\alpha_B+1/\alpha_N\,}\approx 0.054",
   "The compute exponent is not free — it is derived from the N, S and B exponents. The prediction (0.054) matches the direct fit (0.050), a genuine consistency check that the whole framework hangs together.",
   [("α_N,α_S,α_B","the three base exponents","0.076 / 0.76 / 0.21")]),
 ], "alloc", "N (amber) climbs far faster than D (blue); S (green) is nearly flat."),

 ("Where the laws must break", "§6.3 · Contradiction", [
  ("6.7","single-epoch data usage",
   r"D(C_{\min})=\frac{2\,C_{\min}}{6\,N(C_{\min})}\approx(4\times10^{10})\,C_{\min}^{0.26}\ \text{tokens}",
   "The most data compute-efficient training can consume without reusing any (one epoch). It grows so slowly that a big model eventually starves for fresh data relative to its size.",
   [("2C_min","train at critical batch, C = 2·C_min","—")]),
  ("6.8","the breakdown point",
   r"C^{*}\!\sim\!10^{4}\,\text{PF-days},\ \ N^{*}\!\sim\!10^{12},\ \ D^{*}\!\sim\!10^{12},\ \ L^{*}\!\sim\!1.7\ \text{nats}",
   "Where the compute law L(C_min) dips below the data law L(D) — a contradiction. The paper conjectures this crossing is roughly where Transformers reach maximal performance and the power laws must bend. The numbers are uncertain by an order of magnitude.",
   [("L*","conjectured entropy-per-token estimate","≈ 1.7 nats")]),
 ], "breakdown", "L(C_min) amber vs. the data-limited floor L(D(C_min)) blue — they cross near 10⁴ PF-days."),
]

def params_html(ps):
    rows="".join(
      f'<tr><td class="sym">$${s}$$</td><td>{d}</td><td class="val">{v}</td></tr>' for s,d,v in ps)
    return f'<table class="params"><thead><tr><th>symbol</th><th>meaning</th><th>value</th></tr></thead><tbody>{rows}</tbody></table>'

def section_html(title, kicker, eqs, fig, cap):
    cards=""
    for num,name,latex,does,ps in eqs:
        cards+=f'''
        <div class="eqcard">
          <div class="eqhead"><span class="eqname">{name}</span><span class="eqno">({num})</span></div>
          <div class="eq">$${latex}$$</div>
          <div class="does"><b>What it does.</b> {does}</div>
          {params_html(ps)}
        </div>'''
    figblock=""
    if fig:
        figblock=f'<figure><img src="{FIG[fig]}"><figcaption>{cap}</figcaption></figure>'
    return f'''
    <section>
      <div class="skicker">{kicker}</div>
      <h2>{title}</h2>
      {cards}
      {figblock}
    </section>'''

body="".join(section_html(*s) for s in SECTIONS)

HTML=f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Scaling Laws — the math explained</title>
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
  /* cover */
  .cover{{height:250mm;display:flex;flex-direction:column;justify-content:center;
    background:radial-gradient(600px 300px at 80% 10%,#17223c 0%,transparent 60%),
    radial-gradient(500px 260px at 0% 100%,#1a1430 0%,transparent 55%),var(--ink);
    border:1px solid var(--line);border-radius:14px;padding:30mm 22mm;page-break-after:always}}
  .cover .tag{{font-family:"Consolas",monospace;font-size:11px;letter-spacing:.3em;color:var(--amber);text-transform:uppercase}}
  .cover h1{{font-size:46px;line-height:1.03;letter-spacing:-.02em;margin:14px 0 8px;font-weight:800}}
  .cover h1 .g{{background:linear-gradient(92deg,var(--amber),#FFC85C 45%,var(--cool));
    -webkit-background-clip:text;background-clip:text;color:transparent}}
  .cover .sub{{font-size:16px;color:var(--mut);line-height:1.6;max-width:150mm}}
  .cover .meta{{margin-top:26px;font-family:"Consolas",monospace;font-size:12px;color:var(--mut2);line-height:1.9}}
  .cover .meta b{{color:var(--txt)}}
  .cover .rule{{height:2px;width:70mm;background:linear-gradient(90deg,var(--amber),transparent);margin:18px 0}}
  /* sections */
  section{{page-break-inside:avoid;margin:0 0 14px}}
  .skicker{{font-family:"Consolas",monospace;font-size:10px;letter-spacing:.22em;text-transform:uppercase;color:var(--amber);margin-top:6px}}
  h2{{font-size:22px;letter-spacing:-.01em;margin:2px 0 12px;font-weight:700;border-bottom:1px solid var(--line);padding-bottom:7px}}
  .eqcard{{background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);
    border-left:3px solid var(--amber);border-radius:11px;padding:12px 15px;margin:0 0 11px;page-break-inside:avoid}}
  .eqhead{{display:flex;justify-content:space-between;align-items:baseline}}
  .eqname{{font-family:"Consolas",monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--amber)}}
  .eqno{{font-family:"Consolas",monospace;font-size:11px;color:var(--mut2)}}
  .eq{{text-align:center;padding:8px 4px 10px;font-size:17px}}
  .does{{font-size:12px;line-height:1.6;color:var(--mut);margin-bottom:9px}}
  .does b{{color:var(--txt)}}
  table.params{{width:100%;border-collapse:collapse;font-size:11px}}
  table.params th{{text-align:left;font-family:"Consolas",monospace;font-size:8.5px;letter-spacing:.1em;
    text-transform:uppercase;color:var(--mut2);border-bottom:1px solid var(--line);padding:3px 8px}}
  table.params td{{padding:4px 8px;border-bottom:1px solid #1C2333;color:var(--mut);vertical-align:top}}
  table.params td.sym{{color:var(--amber);width:60px}}
  table.params td.val{{color:var(--good);font-family:"Consolas",monospace;text-align:right;white-space:nowrap;width:90px}}
  figure{{margin:8px 0 4px;text-align:center;page-break-inside:avoid}}
  figure img{{width:100%;max-width:172mm;border:1px solid var(--line);border-radius:10px}}
  figcaption{{font-family:"Consolas",monospace;font-size:9.5px;color:var(--mut2);margin-top:5px}}
  .foot{{margin-top:10px;border-top:1px solid var(--line);padding-top:8px;
    font-family:"Consolas",monospace;font-size:9px;color:var(--mut2);line-height:1.7}}
</style></head>
<body>
  <div class="cover">
    <div class="tag">Paper → Code · arXiv:2001.08361</div>
    <h1>Scaling Laws<br><span class="g">for Neural Language Models</span></h1>
    <div class="rule"></div>
    <div class="sub">Every equation of Kaplan, McCandlish et&nbsp;al. (OpenAI, 2020), extracted and
      explained parameter by parameter — with the fitted constants and a live-computed plot for each law.
      Test loss L is the cross-entropy in nats/token on WebText2.</div>
    <div class="meta">
      <b>The math explained</b> — companion to the interactive site<br>
      Original authors: J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, et al.<br>
      Visualised &amp; annotated by <b>Abdulsamad Teniola Muyideen</b><br>
      13 core equations · Tables 4–6 constants · 6 figures
    </div>
  </div>
  {body}
  <div class="foot">
    Scaling Laws for Neural Language Models — the math explained · constants from Tables 5 &amp; 6 of arXiv:2001.08361 ·
    figures computed directly from the fitted power laws · © Abdulsamad Teniola Muyideen · companion to the interactive "Scaling Laws, Animated" site.
  </div>
  <script>
    document.addEventListener("DOMContentLoaded",function(){{
      renderMathInElement(document.body,{{delimiters:[{{left:"$$",right:"$$",display:false}}],throwOnError:false}});
      document.title="rendered";
    }});
  </script>
</body></html>'''

open("_pdf.html","w",encoding="utf-8").write(HTML)
print("wrote _pdf.html", len(HTML), "bytes")
