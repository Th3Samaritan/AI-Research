# Assemble the print-quality HTML for the Chinchilla PDF (headless Chrome).
import base64

def b64(p):
    with open(p,"rb") as f: return "data:image/png;base64,"+base64.b64encode(f.read()).decode()
FIG={k:b64(f"fig_{k}.png") for k in ["law","isoflop","frontier","penalty"]}

SECTIONS=[
 ("The parametric loss law","§3.3 · Approach 3",[
  ("2 / 10","L(N,D) — three terms explain everything",
   r"\hat{L}(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}=1.69+\frac{406.4}{N^{0.34}}+\frac{410.7}{D^{0.28}}",
   "Predicts the final training loss of a Transformer with N parameters trained on D tokens (single epoch). A classical risk decomposition: E is the Bayes risk — the entropy of natural text, which no model of any size beats; A/N^α is the capacity error of a finite network; B/D^β is the estimation error of finite data seen once. Fit to over 400 DeepMind training runs.",
   [("E","irreducible loss floor (entropy of text)","1.69 nats"),
    ("A, α","capacity term scale and exponent","406.4, 0.34"),
    ("B, β","data term scale and exponent","410.7, 0.28"),
    ("N, D","parameters; training tokens","—")]),
  ("3 / 11","the fitting objective",
   r"\min_{a,b,e,\alpha,\beta}\;\sum_{\text{runs }i}\mathrm{Huber}_{\delta}\Big(\mathrm{LSE}\big(a-\alpha\log N_i,\;b-\beta\log D_i,\;e\big)-\log L_i\Big)",
   "How (A,B,E,α,β) are estimated: minimise a Huber loss (δ=10⁻³) between predicted and observed log-loss with L-BFGS, from a grid of initialisations. Huber = robust to outliers, so the noisy low-compute runs don't drag the fit; log-space keeps A=e^a, B=e^b, E=e^e positive. Runnable at tiny scale in Nanogpt/nano_chinchilla.py.",
   [("LSE","log-sum-exp operator","—"),("δ","Huber threshold","10⁻³")]),
 ],"law","Loss vs tokens for several model sizes — every curve floors at its capacity limit; nothing beats E=1.69."),

 ("IsoFLOP profiles","§3.2 · Approach 2",[
  ("§3.2","the valley experiment",
   r"\mathrm{IsoL}(N;C)=\hat{L}\!\left(N,\tfrac{C}{6N}\right),\qquad C=6ND\ \text{fixed}",
   "The question a lab actually faces: given C FLOPs, what model should I train? Fix the budget, sweep model size — every extra parameter costs tokens (D = C/6N). Too small a model can't represent the distribution; too big a model is undertrained. The paper ran 9 budgets (6×10¹⁸ → 3×10²¹ FLOPs), fit a parabola to each valley in log N, then a power law through the minima. The crucial methodological fix over Kaplan 2020: the cosine learning-rate schedule is stretched to match each run's token horizon — reusing one schedule for all horizons (as Kaplan did) under-credits the long-data runs and inflates the model-size exponent.",
   [("C","training FLOPs, C ≈ 6·N·D","—"),
    ("N_opt(C)","valley bottom — optimal model at budget C","—")]),
 ],"isoflop","IsoFLOP valleys computed from the fitted law; dots mark N_opt(C) — the bottoms line up in log-log space."),

 ("The compute-optimal frontier","Eq. 4 · closed form",[
  ("4","optimal allocation",
   r"N_{opt}(C)=G\left(\tfrac{C}{6}\right)^{a}\!,\;\; D_{opt}(C)=G^{-1}\left(\tfrac{C}{6}\right)^{b}\!,\;\; G=\left(\tfrac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}\!,\;\; a=\tfrac{\beta}{\alpha+\beta},\; b=\tfrac{\alpha}{\alpha+\beta}",
   "Minimise L(N,D) subject to 6ND = C and the optimum falls out in closed form. With the fitted α, β: a ≈ 0.46, b ≈ 0.54 — model size and data scale in near-equal proportion with compute. Approaches 1 and 2 give a = b ≈ 0.50 independently. All three contradict Kaplan's a = 0.73, b = 0.27.",
   [("a","N_opt exponent (three approaches)","0.50 / 0.49 / 0.46"),
    ("b","D_opt exponent","0.50 / 0.51 / 0.54"),
    ("Kaplan","the claim being corrected","a=0.73, b=0.27")]),
  ("Table 3","the 20-tokens-per-parameter rule",
   r"D_{opt}\;\approx\;20\times N_{opt}",
   "With a = b = 0.5 (Approaches 1–2) the ratio D/N is a constant, and Table 3 pins it near 20 across every practical budget: 400M→8B tokens, 1B→20.2B, 10B→205B, 67B→1.5T, 175B→3.7T. Chinchilla itself: 1.4T/70B = 20 exactly. Subtlety: the Approach-3 closed form has a < b, so its ratio drifts upward (~40–60 in range) — which is why Approach 3 predicts the smallest optimal models of the three.",
   [("D/N","frontier ratio, Approaches 1–2","≈ 20"),
    ("D/N","frontier ratio, Approach 3","~40–60, slowly growing")]),
 ],"frontier","The frontier vs the Kaplan rule, with real models plotted: Chinchilla on the line; GPT-3, Gopher, MT-NLG far above it."),

 ("The verdict","§3.4 & §4 · Chinchilla vs Gopher",[
  ("§4","proof by training",
   r"\text{Chinchilla: }70\text{B}\times 1.4\text{T tokens}\;\;\text{vs}\;\;\text{Gopher: }280\text{B}\times 300\text{B tokens}\;\;(\text{same }C\approx 5.8\times10^{23})",
   "The paper's hypothesis test: spend Gopher's exact budget the compute-optimal way — 4× smaller, 4× more data. Chinchilla outperforms Gopher (and GPT-3, Jurassic-1, MT-NLG) essentially everywhere: +7.6% average on MMLU (67.5%), better bits-per-byte on all 19 Pile subsets tested, better on the vast majority of BIG-bench tasks — while being 4× cheaper at inference and fine-tuning forever after. Given ~10²⁵ FLOPs, the optimal Gopher-class model would be ~280B on 6.8T tokens; a 1T-parameter model is not optimal below ~10²⁶ FLOPs.",
   [("MMLU","Chinchilla 67.5% vs Gopher 60.0%","+7.5 pts"),
    ("inference","cost ratio vs Gopher","4× cheaper")]),
 ],"penalty","Feeding both allocation rules into the same loss surface: the Kaplan-rule penalty compounds with compute."),
]

def params_html(ps):
    rows="".join(f'<tr><td class="sym">$${s}$$</td><td>{d}</td><td class="val">{v}</td></tr>' for s,d,v in ps)
    return f'<table class="params"><thead><tr><th>symbol</th><th>meaning</th><th>value</th></tr></thead><tbody>{rows}</tbody></table>'

def section_html(title,kicker,eqs,fig,cap):
    cards=""
    for num,name,latex,does,ps in eqs:
        cards+=f'''<div class="eqcard"><div class="eqhead"><span class="eqname">{name}</span><span class="eqno">({num})</span></div>
        <div class="eq">$${latex}$$</div><div class="does"><b>What it does.</b> {does}</div>{params_html(ps)}</div>'''
    figblock=f'<figure><img src="{FIG[fig]}"><figcaption>{cap}</figcaption></figure>' if fig else ""
    return f'<section><div class="skicker">{kicker}</div><h2>{title}</h2>{cards}{figblock}</section>'

body="".join(section_html(*s) for s in SECTIONS)

HTML=f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Chinchilla — the math explained</title>
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
  .cover h1{{font-size:44px;line-height:1.05;letter-spacing:-.02em;margin:14px 0 8px;font-weight:800}}
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
  table.params td.sym{{color:var(--amber);width:70px}}
  table.params td.val{{color:var(--good);font-family:"Consolas",monospace;text-align:right;white-space:nowrap;width:120px}}
  figure{{margin:8px 0 4px;text-align:center;page-break-inside:avoid}}
  figure img{{width:100%;max-width:172mm;border:1px solid var(--line);border-radius:10px}}
  figcaption{{font-family:"Consolas",monospace;font-size:9.5px;color:var(--mut2);margin-top:5px}}
  .foot{{margin-top:10px;border-top:1px solid var(--line);padding-top:8px;
    font-family:"Consolas",monospace;font-size:9px;color:var(--mut2);line-height:1.7}}
</style></head>
<body>
  <div class="cover">
    <div class="tag">Paper → Code · arXiv:2203.15556</div>
    <h1>Chinchilla<br><span class="g">Training Compute-Optimal LLMs</span></h1>
    <div class="rule"></div>
    <div class="sub">Every equation of Hoffmann et&nbsp;al. (DeepMind, 2022), extracted and explained
      parameter by parameter — the parametric loss law, the IsoFLOP experiment, the closed-form frontier,
      and the correction of Kaplan's scaling laws that shrank every model after it.
      Loss L in nats/token on MassiveText; C = 6·N·D FLOPs.</div>
    <div class="meta">
      <b>The math explained</b> — companion to the interactive site<br>
      Original authors: J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, et al.<br>
      Visualised &amp; annotated by <b>Abdulsamad Teniola Muyideen</b><br>
      Core equations 2–4 &amp; 10–11 · Tables 2–3 · 4 computed figures · runnable at Nanogpt/nano_chinchilla.py
    </div>
  </div>
  {body}
  <div class="foot">
    Chinchilla — the math explained · constants from Eq. 10 &amp; Tables 2–3 of arXiv:2203.15556 ·
    figures computed directly from the fitted law · © Abdulsamad Teniola Muyideen · sister document: "Scaling Laws — the math explained" (Kaplan 2020).
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
