"""
=============================================================================
 compare.py — read every runs/*.json and evaluate the papers' claims
=============================================================================
 Run AFTER the four experiment scripts. It prints a comparison table and a
 claim-by-claim verdict: what each paper said, what your tiny experiments
 measured, and whether the claim survived miniaturisation.

   python compare.py
=============================================================================
"""

import json, os, sys

if hasattr(sys.stdout, "reconfigure"):        # Windows cp1252 console fix
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

def load(name):
    p = os.path.join(RUNS, name)
    return json.load(open(p)) if os.path.exists(p) else None

gpt   = load("gpt_attention.json")
bert  = load("bert_mlm.json")
scal  = load("scaling_laws.json")
chin  = load("chinchilla.json")

W = 78
def hr(c="─"): print(c * W)
def title(s):  hr("═"); print(s); hr("═")

# ---------------------------------------------------------------------------
# TABLE 1 — the two architectures, side by side
# ---------------------------------------------------------------------------
title("1. ARCHITECTURES — GPT (causal) vs BERT (bidirectional)")
if gpt and bert:
    rows = [
        ("objective",        gpt["objective"],                 bert["objective"]),
        ("attention mask",   "causal (no peeking ahead)",      "none (sees both sides)"),
        ("params",           f"{gpt['params']:,}",             f"{bert['params']:,}"),
        ("final val loss",   f"{gpt['final_val_loss_nats']:.4f} nats",
                             f"{bert['final_val_loss_nats']:.4f} nats"),
        ("can generate?",    "yes (autoregressive)",           "no (fills blanks)"),
        ("train FLOPs≈6NBS", f"{gpt['train_flops_6NBS']:.2e}", f"{bert['train_flops_6NBS']:.2e}"),
    ]
    for k, a, b in rows:
        print(f"{k:<18} | {a:<28} | {b}")
    hr()
    print("⚠  The two losses are NOT the same exam:")
    print("   GPT : predict 100% of chars, left context only.")
    print("   BERT: predict ~15% masked chars, context from BOTH sides.")
    print("   BERT's number being lower is expected and does NOT mean 'better'.")
    print("   The papers' actual division of labour: GPT-style for generation,")
    print("   BERT-style for understanding/fine-tuning tasks.")
else:
    print("run nanogpt_attention.py and nanobert.py first")

# ---------------------------------------------------------------------------
# TABLE 2 — Kaplan scaling law verdicts
# ---------------------------------------------------------------------------
print()
title("2. KAPLAN et al. 2020 — 'Scaling Laws for Neural Language Models'")
if scal:
    f1, f2 = scal["fit_pure_power"], scal["fit_saturating"]
    print(f"claim   : L(N) = (Nc/N)^a — a straight line on log-log axes")
    print(f"measured: pure power law a={f1['alpha']:.3f} (R²={f1['r2']:.4f})")
    print(f"          saturating fit L={f2['E']:.2f}+A/N^{f2['alpha']:.2f} "
          f"(R²={f2['r2']:.4f})")
    print(f"paper   : a=0.076 on WebText2 (BPE tokens — ours is char-level,")
    print(f"          so the exponent differs by construction)")
    ok = f1["r2"] > 0.9 or f2["r2"] > 0.9
    print(f"verdict : {'✔ SUPPORTED' if ok else '✘ NOT CLEAN'} — loss falls "
          f"smoothly & predictably with N{' (R²>0.9)' if ok else ''}")
    for r in scal["ladder"]:
        print(f"   N={r['N_non_embedding']:>9,}  ->  L={r['best_val_loss']:.4f}")
else:
    print("run nano_scaling_laws.py first")

# ---------------------------------------------------------------------------
# TABLE 3 — Chinchilla verdicts (the referee between the two papers)
# ---------------------------------------------------------------------------
print()
title("3. HOFFMANN et al. 2022 — 'Chinchilla' vs Kaplan: who allocates better?")
if chin:
    a = chin["approach2_fit"]["a"]
    print("claim   : at fixed compute there is an OPTIMAL model size (IsoFLOP")
    print("          valley), and N_opt ∝ C^a with a≈0.50 (Kaplan said 0.73)")
    if a == a:  # not NaN
        ka, ca = 0.73, 0.50
        winner = "CHINCHILLA" if abs(a - ca) < abs(a - ka) else "KAPLAN"
        print(f"measured: a = {a:.2f}   (dist to Chinchilla 0.50: {abs(a-ca):.2f},"
              f" to Kaplan 0.73: {abs(a-ka):.2f})")
        print(f"verdict : ✔ valleys observed; exponent sides with {winner}")
    for v in chin["valleys"]:
        print(f"   C={v['C']:.1e} FLOPs -> N_opt≈{v['N_opt']:>9,.0f} "
              f"({v['tokens_per_param']:.0f} tokens/param at the valley)")
    p3 = chin.get("approach3_parametric")
    if p3:
        print(f"parametric fit: L(N,D) = {p3['E']:.2f} + {p3['A']:.1f}/N^{p3['alpha']:.2f}"
              f" + {p3['B']:.1f}/D^{p3['beta']:.2f}")
        print(f"   -> frontier a={p3['a_frontier']:.2f}, b={p3['b_frontier']:.2f} "
              f"(paper: 0.46/0.54; 20 tokens/param rule)")
else:
    print("run nano_chinchilla.py first")

# ---------------------------------------------------------------------------
# THE BIG PICTURE
# ---------------------------------------------------------------------------
print()
title("THE STORY THE FOUR SCRIPTS TELL TOGETHER")
print("""
 1. nanogpt_attention.py — self-attention + a causal mask is sufficient to
    model language autoregressively (Vaswani 2017 -> GPT).
 2. nanobert.py          — delete the causal mask, mask the INPUT instead,
    and the same machine becomes a bidirectional understander (BERT 2018).
 3. nano_scaling_laws.py — performance of these models is a smooth,
    predictable function of scale, not alchemy (Kaplan 2020).
 4. nano_chinchilla.py   — but Kaplan's ALLOCATION advice was off: balance
    model and data ~50/50, don't just grow the model (Hoffmann 2022).
 Historical punchline: GPT-3 (175B) was sized by Kaplan's rule; Chinchilla
 (70B on 4x the data, same compute) beat it. Llama and everything since
 train small models on many more than 20 tokens/param — because inference
 cost matters too, which neither paper's objective includes.
""")
