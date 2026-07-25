"""
AI4DFT Lesson 3 — The Variational Principle & Basis Sets
========================================================

Run me with:   python lesson03_variational_basis_sets.py

THE VARIATIONAL PRINCIPLE (the most important theorem in this course):
For ANY trial wavefunction psi_trial, the energy you compute from it is
GREATER THAN OR EQUAL to the true ground-state energy:

    E[psi_trial] >= E_exact        (equality only for the true ground state)

This means "guess and minimize" is a rigorous strategy: the lower your
energy, the better your wavefunction — you can never overshoot below truth.
Hartree-Fock, DFT, and quantum chemistry are all built on this.

BASIS SETS: instead of a wavefunction on a grid, expand it in a handful of
fixed functions (here: Gaussians, exp(-a r^2), the industry standard because
their integrals have closed forms):

    psi(r) = sum_i  c_i * g_i(r)

Minimizing the energy over the coefficients c_i turns the Schrödinger
equation into a small GENERALIZED matrix eigenvalue problem:

    H c = E S c      (S = overlap matrix, because Gaussians aren't orthogonal)

We test on hydrogen (exact E = -0.5 hartree):
  Part A: one Gaussian, scan its width, find the best -> E = -0.4244 (not great!)
  Part B: 1, 2, 3 Gaussians -> watch the energy march DOWN toward -0.5
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pathlib import Path

# Closed-form integrals for s-type Gaussians g_i = exp(-a_i r^2), hydrogen (Z=1)
# (standard results — see e.g. Szabo & Ostlund, "Modern Quantum Chemistry")
def overlap(a, b):   return (np.pi / (a + b)) ** 1.5
def kinetic(a, b):   return 3.0 * a * b * np.pi**1.5 / (a + b) ** 2.5
def nuclear(a, b):   return -2.0 * np.pi / (a + b)          # Z = 1


def variational_energy(exponents):
    """Lowest generalized eigenvalue of H c = E S c for a Gaussian basis."""
    a = np.asarray(exponents, dtype=float)
    A, B = np.meshgrid(a, a, indexing="ij")
    S = overlap(A, B)
    H = kinetic(A, B) + nuclear(A, B)
    E, C = eigh(H, S)               # generalized eigenproblem
    return E[0], C[:, 0]


# ---------------------------------------------------------------------------
# Part A — a single Gaussian: scan the exponent alpha
#   Analytic: E(alpha) = 3 alpha/2 - 2 sqrt(2 alpha / pi)
#   Best possible single Gaussian: alpha = 8/(9 pi), E = -4/(3 pi) = -0.4244
# ---------------------------------------------------------------------------
alphas = np.linspace(0.02, 2.0, 400)
E_curve = np.array([variational_energy([a])[0] for a in alphas])
i_best = np.argmin(E_curve)

print("Part A — one Gaussian for hydrogen:")
print(f"  best exponent alpha = {alphas[i_best]:.4f}  (analytic: 8/9pi = {8/(9*np.pi):.4f})")
print(f"  best energy   E     = {E_curve[i_best]:.4f}  (analytic: -4/3pi = {-4/(3*np.pi):.4f})")
print(f"  exact               = -0.5000")
print("  ==> Even the BEST single Gaussian misses by ~15%. Gaussians have the")
print("      wrong shape at the nucleus (no cusp) and wrong decay at long range.")

# ---------------------------------------------------------------------------
# Part B — add more Gaussians: the energy can only go DOWN (variational!)
# ---------------------------------------------------------------------------
bases = {
    "1 Gaussian": [8 / (9 * np.pi)],
    "2 Gaussians": [0.202, 1.33],
    "3 Gaussians (STO-3G)": [3.42525091, 0.62391373, 0.16885540],
}
print("\nPart B — basis-set convergence (exact = -0.5):")
results = {}
for name, exps in bases.items():
    E0, _ = variational_energy(exps)
    results[name] = E0
    print(f"  {name:22s} E = {E0:.5f}   error = {abs(E0 + 0.5):.5f}")
print("  ==> This is why real calculations quote a BASIS SET. More basis")
print("      functions = lower (better) energy, at higher cost. Plane-wave DFT")
print("      codes control the same trade-off with one knob: the energy cutoff.")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.suptitle("Lesson 3 — Variational principle: energies approach the truth from above",
             fontsize=13, fontweight="bold")

ax1.plot(alphas, E_curve, lw=2, color="#1f77b4", label="E(α), one Gaussian")
ax1.axhline(-0.5, color="#2ca02c", lw=1.5, ls="--", label="exact  (-0.5)")
ax1.plot(alphas[i_best], E_curve[i_best], "o", ms=8, color="#d62728", label="best α")
ax1.set_title("A — scan one Gaussian's width")
ax1.set_xlabel("exponent α"); ax1.set_ylabel("energy (hartree)")
ax1.legend(frameon=False)

names = list(results)
vals = [results[n] for n in names]
ax2.bar(range(len(vals)), vals, color=["#9ecae1", "#4292c6", "#08519c"], width=0.55)
ax2.axhline(-0.5, color="#2ca02c", lw=1.5, ls="--", label="exact (-0.5)")
ax2.set_xticks(range(len(vals)))
ax2.set_xticklabels(["1G", "2G", "3G\n(STO-3G)"])
ax2.set_ylim(-0.52, 0)
ax2.set_title("B — more basis functions → closer to truth (never below!)")
ax2.set_ylabel("energy (hartree)")
ax2.legend(frameon=False)

for ax in (ax1, ax2):
    ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = Path(__file__).parent / "lesson03_variational.png"
fig.savefig(out, dpi=150)
print(f"\nSaved figure to {out}")
plt.show()
