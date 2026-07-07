/* ============================================================================
   Training Compute-Optimal Large Language Models ("Chinchilla")
   Hoffmann et al., DeepMind 2022 — arXiv:2203.15556
   Complete, documented implementation of every core formula.

   Loss L is cross-entropy in nats/token on MassiveText.
   Compute C is in FLOPs, related to model/data by  C ≈ 6·N·D.
   ==========================================================================*/

/* ---------------------------------------------------------------------------
   FITTED CONSTANTS (Eq. 10, Appendix D.2) — the parametric loss fit
   -------------------------------------------------------------------------*/
const CH = {
  E:     1.69,    // entropy of natural text — the irreducible loss floor
  A:     406.4,   // scale of the model-capacity term
  B:     410.7,   // scale of the finite-data term
  alpha: 0.34,    // how fast capacity error falls with N
  beta:  0.28,    // how fast data error falls with D

  // Table 2 — frontier exponents N_opt ∝ C^a, D_opt ∝ C^b per approach
  approaches: {
    a1: { a: 0.50, b: 0.50, name: "Minimum over training curves" },
    a2: { a: 0.49, b: 0.51, name: "IsoFLOP profiles" },
    a3: { a: 0.46, b: 0.54, name: "Parametric loss fit" },
    kaplan: { a: 0.73, b: 0.27, name: "Kaplan et al. 2020" },
  },

  // Landmark models (params, training tokens) for context
  models: {
    Chinchilla: { N: 70e9,  D: 1.4e12 },
    Gopher:     { N: 280e9, D: 300e9  },
    GPT3:       { N: 175e9, D: 300e9  },
    MTNLG:      { N: 530e9, D: 270e9  },
  },
};

/* ===========================================================================
   (Eq. 2 / 10)  L(N,D) = E + A/N^alpha + B/D^beta
   ---------------------------------------------------------------------------
   WHAT IT DOES: Predicts the final training loss of a Transformer with N
   parameters trained on D tokens (single epoch). A classical risk
   decomposition in three readable terms:
     E        — the Bayes risk: what a PERFECT model would score; the
                entropy of natural text itself. No model beats E.
     A/N^α    — function-approximation error: a finite net of size N cannot
                represent the true distribution. Shrinks as N grows.
     B/D^β    — optimisation/estimation error: finitely many tokens, seen
                once. Shrinks as D grows.
   PARAMETERS: N params (non-embedding-ish, full params in the paper),
               D tokens. Fitted: E=1.69, A=406.4, B=410.7, α=0.34, β=0.28.
   ========================================================================= */
function L_ND(N, D) {
  return CH.E + CH.A / Math.pow(N, CH.alpha) + CH.B / Math.pow(D, CH.beta);
}

/* ===========================================================================
   C = 6·N·D  — the compute identity both papers share
   ---------------------------------------------------------------------------
   WHAT IT DOES: Total training FLOPs ≈ 6 FLOPs per parameter per token
   (2 forward + 4 backward, multiply-accumulate counted as 2).
   Given any two of (C, N, D) you get the third:
   ========================================================================= */
function C_of(N, D)  { return 6 * N * D; }
function D_given(C, N) { return C / (6 * N); }

/* ===========================================================================
   (Eq. 3 / 11)  The fitting objective (documented for completeness)
   ---------------------------------------------------------------------------
   min over (a,b,e,α,β) of  Σ_runs Huber_δ( LSE(a−α·logN, b−β·logD, e) − logL )
   where LSE = log-sum-exp and A=e^a, B=e^b, E=e^e; δ=1e-3.
   WHY HUBER: robust to outliers — the low-compute runs are noisy and the
   fit should not chase them. WHY LOG-SPACE: keeps A,B,E positive and makes
   errors multiplicative, matching how loss differences matter.
   (Implemented runnable at tiny scale in Nanogpt/nano_chinchilla.py.)
   ========================================================================= */

/* ===========================================================================
   (Eq. 4)  The closed-form compute-optimal frontier
   ---------------------------------------------------------------------------
     N_opt(C) = G · (C/6)^a      D_opt(C) = G⁻¹ · (C/6)^b
     G = (αA / βB)^(1/(α+β)),    a = β/(α+β),    b = α/(α+β)
   WHAT IT DOES: Minimise L(N,D) subject to 6ND = C (Lagrange on logs) and
   the optimal split falls out in closed form. With the fitted α=0.34,
   β=0.28:  a = 0.28/0.62 ≈ 0.46 and b = 0.34/0.62 ≈ 0.54 — model size and
   data should grow in nearly EQUAL proportion with compute.
   Contrast Kaplan 2020: a=0.73, b=0.27 (grow the model, barely the data).
   ========================================================================= */
function G_const() {
  return Math.pow((CH.alpha * CH.A) / (CH.beta * CH.B), 1 / (CH.alpha + CH.beta));
}
function a_frontier() { return CH.beta  / (CH.alpha + CH.beta); }  // ≈ 0.46
function b_frontier() { return CH.alpha / (CH.alpha + CH.beta); }  // ≈ 0.54
function N_opt(C) { return G_const()     * Math.pow(C / 6, a_frontier()); }
function D_opt(C) { return (1/G_const()) * Math.pow(C / 6, b_frontier()); }

/* ===========================================================================
   The "20 tokens per parameter" rule of thumb — and where it comes from
   ---------------------------------------------------------------------------
   SUBTLETY WORTH KNOWING: the famous "≈20 tokens/param" comes from
   Approaches 1 & 2, where a = b = 0.5 exactly. Equal exponents make the
   ratio D_opt/N_opt a CONSTANT, and Table 3 pins that constant near 20
   (Chinchilla itself: 1.4T/70B = 20).
   The Approach-3 closed form above has a ≈ 0.46 < b ≈ 0.54, so ITS ratio
   grows slowly with C (≈40-60 in the practical range) — that is why
   Approach 3 predicts slightly smaller models than the other two.
   ========================================================================= */
function tokensPerParam(C) { return D_opt(C) / N_opt(C); }   // Approach-3 ratio

// Approach 1/2 frontier: a=b=0.5 with the D=20·N calibration from Table 3.
//   6·N·(20N) = C  =>  N_opt = sqrt(C/120),  D_opt = 20·N_opt.
function N_opt12(C) { return Math.sqrt(C / 120); }
function D_opt12(C) { return 20 * N_opt12(C); }

/* ===========================================================================
   Loss along the frontier  L_opt(C) = L(N_opt(C), D_opt(C))
   ---------------------------------------------------------------------------
   The best loss any allocation can reach at budget C — the envelope that
   every IsoFLOP valley bottom traces out.
   ========================================================================= */
function L_opt(C) { return L_ND(N_opt(C), D_opt(C)); }

/* ===========================================================================
   Kaplan-rule allocation, for the head-to-head comparison
   ---------------------------------------------------------------------------
   Same budget C, but split Kaplan's way (N ∝ C^0.73 anchored so both rules
   agree at C0 = Gopher-scale anchor). Feeding this N into the SAME loss
   surface L(N,D) with D = C/6N shows the loss penalty of over-sizing.
   ========================================================================= */
function N_kaplan(C, C0) {
  C0 = C0 || 5.76e23;                       // anchor: Gopher's budget
  return N_opt(C0) * Math.pow(C / C0, 0.73);
}
function L_kaplan(C) { const N = N_kaplan(C); return L_ND(N, D_given(C, N)); }

/* ===========================================================================
   IsoFLOP profile — the paper's Figure 3 as a function
   ---------------------------------------------------------------------------
   At fixed budget C, loss as a function of model size N (data implied):
     IsoL(N; C) = L(N, C/6N)
   Too-small N: capacity term dominates. Too-big N: data term dominates
   (undertrained). The minimum is N_opt(C) — the valley.
   ========================================================================= */
function isoL(N, C) { return L_ND(N, D_given(C, N)); }

// Expose for the page.
if (typeof window !== "undefined") {
  window.CH = CH;
  window.CHfn = { L_ND, C_of, D_given, G_const, a_frontier, b_frontier,
                  N_opt, D_opt, N_opt12, D_opt12, tokensPerParam,
                  L_opt, N_kaplan, L_kaplan, isoL };
}
