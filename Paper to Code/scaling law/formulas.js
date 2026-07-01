/* ============================================================================
   Scaling Laws for Neural Language Models  (Kaplan et al., 2020, arXiv:2001.08361)
   Complete, documented implementation of every formula in the paper.

   Each function below is a faithful, runnable version of an equation from the
   paper. The block comment above each one states:
     - the equation number,
     - what it does (the phenomenon it predicts),
     - every parameter and its meaning + units,
     - the fitted numerical values.

   Loss L is the autoregressive cross-entropy in nats/token on WebText2.
   ==========================================================================*/

/* ---------------------------------------------------------------------------
   FITTED CONSTANTS  (Table 5 — the canonical, headline values)
   These are tokenization-dependent scales; the exponents are near-universal.
   -------------------------------------------------------------------------*/
const SL = {
  // L(N): loss vs non-embedding parameter count
  aN:   0.076,        // alpha_N  — parameter power-law exponent (dimensionless)
  Nc:   8.8e13,       // N_c      — parameter scale [non-embedding params]

  // L(D): loss vs dataset size (tokens), early-stopped
  aD:   0.095,        // alpha_D  — data power-law exponent
  Dc:   5.4e13,       // D_c      — data scale [tokens]

  // L(C): naive loss vs compute at fixed batch size
  aC:   0.057,        // alpha_C  — naive compute exponent
  Cc:   1.6e7,        // C_c      — naive compute scale [PF-days]

  // L(Cmin): loss vs optimally-allocated compute (train at B << Bcrit)
  aCmin: 0.050,       // alpha_C^min — optimal-compute exponent
  Ccmin: 3.1e8,       // C_c^min     — optimal-compute scale [PF-days]

  // Bcrit(L): critical batch size vs loss
  aB:   0.21,         // alpha_B  — batch exponent
  Bstar: 2.1e8,       // B*       — batch scale [tokens]

  // L(N,S): loss vs steps (training-time term)
  aS:   0.76,         // alpha_S  — step power-law exponent
  Sc:   2.1e3,        // S_c      — step scale [steps]

  // Compute-efficient frontier (Table 6): X_opt = X_e * Cmin^pX
  pN: 0.73, Ne: 1.3e9,   // optimal model size            [params]
  pB: 0.24, Be: 2.0e6,   // optimal (critical) batch size [tokens]
  pS: 0.03, Se: 5.4e3,   // optimal number of steps       [steps]
  pD: 0.27, De: 2.0e10,  // optimal dataset (1 epoch)     [tokens]
};

/* ===========================================================================
   (1.1)  L(N) = (Nc / N)^alpha_N
   ---------------------------------------------------------------------------
   WHAT IT DOES: Predicts test loss for a model with N non-embedding parameters
   trained to convergence on a sufficiently large dataset (data is NOT the
   bottleneck). Every time you multiply N by a fixed factor, the loss drops by
   a fixed multiplicative factor — a straight line on a log-log plot.
   PARAMETERS:
     N   : number of non-embedding parameters.
     Nc  : parameter scale = 8.8e13 (no fundamental meaning; tokenizer-dependent).
     aN  : exponent = 0.076. Doubling N multiplies loss by 2^-0.076 = 0.95.
   ========================================================================= */
function L_N(N) { return Math.pow(SL.Nc / N, SL.aN); }

/* ===========================================================================
   (1.2)  L(D) = (Dc / D)^alpha_D
   ---------------------------------------------------------------------------
   WHAT IT DOES: Predicts test loss for a large model trained (with early
   stopping) on a limited dataset of D tokens. Data is the bottleneck.
   PARAMETERS:
     D  : dataset size in tokens.
     Dc : data scale = 5.4e13 tokens.
     aD : exponent = 0.095.
   ========================================================================= */
function L_D(D) { return Math.pow(SL.Dc / D, SL.aD); }

/* ===========================================================================
   (1.3)/(6.3)  L(Cmin) = (Cc^min / Cmin)^alpha_C^min
   ---------------------------------------------------------------------------
   WHAT IT DOES: Predicts the best achievable loss for a given amount of
   optimally-allocated compute Cmin (large dataset, optimal model size, batch
   size << Bcrit). This is the trend to use for real compute-budget forecasts.
   PARAMETERS:
     Cmin  : optimal non-embedding compute in PF-days (1 PF-day = 8.64e19 FLOP).
     Ccmin : compute scale = 3.1e8 PF-days.
     aCmin : exponent = 0.050.
   ========================================================================= */
function L_Cmin(Cmin) { return Math.pow(SL.Ccmin / Cmin, SL.aCmin); }

/* ===========================================================================
   (3.3)  L(C) = (Cc / C)^alpha_C     — naive, fixed-batch compute law
   ---------------------------------------------------------------------------
   WHAT IT DOES: Empirical loss vs raw training compute C at FIXED batch size.
   Slightly steeper and messier than the L(Cmin) trend; kept for comparison.
   PARAMETERS: C in PF-days, Cc = 1.6e7 PF-days, aC = 0.057.
   ========================================================================= */
function L_C(C) { return Math.pow(SL.Cc / C, SL.aC); }

/* ===========================================================================
   (1.4)/(5.3)  Bcrit(L) = B* / L^(1/alpha_B)
   ---------------------------------------------------------------------------
   WHAT IT DOES: The critical batch size — the batch size that gives a roughly
   optimal time/compute trade-off. It depends only on the current loss L, NOT
   on model size. It grows (diverges) as loss falls; ~doubles per 13% loss drop.
   PARAMETERS:
     L  : current loss (nats/token).
     B* : batch scale = 2.1e8 tokens.
     aB : exponent = 0.21, so 1/aB ≈ 4.76 (the "L^4.8" seen in the paper).
   RETURNS: critical batch size in tokens.
   ========================================================================= */
function Bcrit(L) { return SL.Bstar / Math.pow(L, 1 / SL.aB); }

/* ===========================================================================
   (1.5)/(4.1)  L(N,D) = [ (Nc/N)^(aN/aD) + Dc/D ]^aD
   ---------------------------------------------------------------------------
   WHAT IT DOES: The single unified law for simultaneous dependence on model
   size N and dataset size D. Governs overfitting: as N grows at fixed D the
   loss stops improving. Reduces to L(N) as D->inf and to L(D) as N->inf.
   PARAMETERS: N params, D tokens; constants aN, aD, Nc, Dc as above.
   ========================================================================= */
function L_ND(N, D) {
  return Math.pow(Math.pow(SL.Nc / N, SL.aN / SL.aD) + SL.Dc / D, SL.aD);
}

/* ===========================================================================
   (4.2)/(4.3)  deltaL(N,D) = L(N,D)/L(N,inf) - 1
                            = [1 + (N/Nc)^(aN/aD) * (Dc/D)]^aD - 1
   ---------------------------------------------------------------------------
   WHAT IT DOES: The *fractional* extra loss caused by overfitting on a finite
   dataset, relative to the infinite-data loss L(N,inf)=L(N). Remarkably it
   depends only on the combination  x = (N^(aN/aD)/Nc^...) * Dc/D  ~ N^0.74 / D.
   PARAMETERS: N params, D tokens.
   ========================================================================= */
function deltaL(N, D) {
  return Math.pow(1 + Math.pow(N / SL.Nc, SL.aN / SL.aD) * (SL.Dc / D), SL.aD) - 1;
}

/* ===========================================================================
   (4.4)  D >~ (5e3) * N^0.74      — no-overfitting data requirement
   ---------------------------------------------------------------------------
   WHAT IT DOES: Minimum dataset size (tokens) needed so overfitting stays
   below the ~0.02 random-seed noise threshold. Data may grow SUB-linearly in
   model size: multiply N by 8 -> multiply D by only ~5.
   PARAMETERS: N params -> returns D in tokens. Exponent 0.74 = 1 - aN/aD.
   ========================================================================= */
function D_noOverfit(N) { return 5e3 * Math.pow(N, 0.74); }

/* ===========================================================================
   (1.6)/(5.6)/(B.1)  L(N,S) = (Nc/N)^aN + (Sc/Smin)^aS
   ---------------------------------------------------------------------------
   WHAT IT DOES: Loss as a function of model size N and number of optimization
   steps Smin (steps measured at the critical batch size) in the infinite-data
   limit. First term = capacity floor; second term = how far training has come.
   PARAMETERS: N params, Smin steps; Sc = 2.1e3 steps, aS = 0.76.
   ========================================================================= */
function L_NS(N, Smin) {
  return Math.pow(SL.Nc / N, SL.aN) + Math.pow(SL.Sc / Smin, SL.aS);
}

/* ===========================================================================
   (5.1)  (S/Smin - 1)(E/Emin - 1) = 1     — steps/examples trade-off
   ---------------------------------------------------------------------------
   WHAT IT DOES: For any target loss, the number of steps S and examples
   processed E=B*S obey this hyperbola. Helper returns steps S required at
   batch size B given the minimum steps Smin and minimum examples Emin.
   Derived: S = Smin * (1 + Emin/(B*Smin)) = Smin*(1 + Bcrit/B).
   ========================================================================= */
function stepsAtBatch(Smin, Bcrit_L, B) { return Smin * (1 + Bcrit_L / B); }

/* ===========================================================================
   (5.4)  Smin(S) = S / (1 + Bcrit(L)/B)
   ---------------------------------------------------------------------------
   WHAT IT DOES: Converts the actual steps S taken at batch size B into the
   *minimum* steps you would have needed at B >> Bcrit (the universal Smin used
   in the training-curve fits).
   PARAMETERS: S steps at batch B (tokens), current loss L.
   ========================================================================= */
function Smin_from_S(S, B, L) { return S / (1 + Bcrit(L) / B); }

/* ===========================================================================
   (5.5)  Cmin(C) = C / (1 + B/Bcrit(L))
   ---------------------------------------------------------------------------
   WHAT IT DOES: Converts raw compute C = 6*N*B*S into the *minimum* compute
   you would have spent at B << Bcrit — the clean quantity used for forecasts.
   PARAMETERS: C in PF-days, batch B (tokens), loss L.
   ========================================================================= */
function Cmin_from_C(C, B, L) { return C / (1 + B / Bcrit(L)); }

/* ===========================================================================
   (5.7)  Sstop(N,D) >~ Sc / [ L(N,D) - L(N,inf) ]^(1/aS)
   ---------------------------------------------------------------------------
   WHAT IT DOES: Lower-bound estimate of the step at which early stopping
   should occur when training is data-limited.
   PARAMETERS: N params, D tokens. Uses L(N,D) and L(N,inf)=L(N).
   ========================================================================= */
function Sstop(N, D) {
  const gap = L_ND(N, D) - L_N(N);
  return SL.Sc / Math.pow(Math.max(gap, 1e-12), 1 / SL.aS);
}

/* ===========================================================================
   (2.1)  N ≈ 12 * n_layer * d_model^2      (non-embedding parameter count)
   full:  N ≈ 2 * d_model * n_layer * (2*d_attn + d_ff)
   ---------------------------------------------------------------------------
   WHAT IT DOES: Counts non-embedding parameters of a decoder Transformer with
   the standard d_attn = d_ff/4 = d_model.
   ========================================================================= */
function N_params(n_layer, d_model) { return 12 * n_layer * d_model * d_model; }

/* ===========================================================================
   (2.2)  C_forward ≈ 2N + 2 * n_layer * n_ctx * d_model     (FLOP/token, fwd)
          C ≈ 6N  (fwd+bwd, ignoring context term)
   ---------------------------------------------------------------------------
   WHAT IT DOES: Compute per token. The training rule of thumb C ≈ 6N per token
   (factor 6 = 2 fwd + 4 bwd multiply-accumulate) underlies C = 6*N*B*S.
   ========================================================================= */
function C_forward(N, n_layer, n_ctx, d_model) {
  return 2 * N + 2 * n_layer * n_ctx * d_model;
}
function C_total(N, B, S) { return 6 * N * B * S; } // FLOP; /8.64e19 -> PF-days

/* ===========================================================================
   (1.7)/(6.1,6.2)/(Table 6)  Compute-efficient frontier
     N_opt   = Ne * Cmin^0.73
     Bcrit_opt = Be * Cmin^0.24
     Smin_opt = Se * Cmin^0.03
     D_opt   = De * Cmin^0.27      (single epoch)
   ---------------------------------------------------------------------------
   WHAT IT DOES: Given an optimal compute budget Cmin (PF-days), these give the
   optimal model size, batch size, number of steps, and dataset size. Almost
   all extra compute should go into a bigger MODEL, not more steps.
   ========================================================================= */
function N_opt(Cmin)  { return SL.Ne * Math.pow(Cmin, SL.pN); }
function B_opt(Cmin)  { return SL.Be * Math.pow(Cmin, SL.pB); }
function S_opt(Cmin)  { return SL.Se * Math.pow(Cmin, SL.pS); }
function D_opt(Cmin)  { return SL.De * Math.pow(Cmin, SL.pD); }

/* ===========================================================================
   (6.4)  alpha_C^min = 1 / (1/aS + 1/aB + 1/aN)   (predicted ≈ 0.054)
   (1.8)  same relation for the optimal-allocation exponent.
   ---------------------------------------------------------------------------
   WHAT IT DOES: Predicts the optimal-compute exponent purely from the N, S, B
   exponents — a non-trivial consistency check that matches the fitted 0.050.
   ========================================================================= */
function alphaCmin_predicted() {
  return 1 / (1 / SL.aS + 1 / SL.aB + 1 / SL.aN);
}

/* ===========================================================================
   (6.7)  D(Cmin) = 2*Cmin / (6 * N(Cmin)) ≈ (4e10) * Cmin^0.26  [tokens]
   (6.8)  Breakdown / "maximal performance" point where L(Cmin) meets L(D):
          C* ~ 1e4 PF-days,  N* ~ 1e12 params,  D* ~ 1e12 tokens,  L* ~ 1.7 nats
   ---------------------------------------------------------------------------
   WHAT IT DOES: The single-epoch data usage of compute-efficient training, and
   the conjectured point at which the power laws must break down.
   ========================================================================= */
function D_epoch(Cmin) { return 4e10 * Math.pow(Cmin, 0.26); }
const BREAKDOWN = { C: 1e4, N: 1e12, D: 1e12, L: 1.7 };

// Expose for the page.
if (typeof window !== 'undefined') {
  window.SL = SL;
  window.SLfn = {
    L_N, L_D, L_C, L_Cmin, Bcrit, L_ND, deltaL, D_noOverfit, L_NS,
    stepsAtBatch, Smin_from_S, Cmin_from_C, Sstop, N_params, C_forward,
    C_total, N_opt, B_opt, S_opt, D_opt, alphaCmin_predicted, D_epoch, BREAKDOWN
  };
}
