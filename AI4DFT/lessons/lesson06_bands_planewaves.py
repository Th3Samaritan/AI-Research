"""
AI4DFT Lesson 6 — Crystals: Band Structures from Plane Waves
============================================================

Run me with:   python lesson06_bands_planewaves.py

A crystal is a PERIODIC potential: V(x + a) = V(x). Bloch's theorem says the
electron states are then labeled by a wavevector k inside the "Brillouin
zone" k in [-pi/a, +pi/a], and the natural basis is PLANE WAVES

    psi_k(x) = sum_G  c_G  exp(i (k+G) x),     G = 2*pi*m/a  (m integer)

— the exact basis used by VASP, Quantum ESPRESSO, Abinit and CASTEP. In this
basis the Hamiltonian for V(x) = V0 * cos(2*pi*x/a) is a tiny matrix:

    H_GG'(k) = 1/2 (k+G)^2 * delta_GG'  +  V0/2 * delta_{G,G' +/- 2pi/a}

Diagonalize it at each k, plot eigenvalues vs k -> the BAND STRUCTURE.

What you'll see (the core mental model of solid-state materials science):
  * V0 = 0 (free electrons): a folded parabola, no gaps -> METAL-like
  * V0 > 0: GAPS open at the zone boundary, width ~ V0. Electrons fill bands
    from the bottom (2 per k-state, spin). Partially filled band -> METAL.
    Filled band + gap to the next -> SEMICONDUCTOR / INSULATOR.
  * This is Lesson 1's double-well splitting scaled up to 10^23 wells:
    discrete levels smear into bands, splittings become gaps.

The "number of plane waves" knob below is EXACTLY the famous plane-wave
energy CUTOFF (ecutwfc) you will set in every real DFT input file.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

a = 1.0                     # lattice constant
M = 7                       # plane waves G = 2*pi*m/a with m = -M..M
G = 2 * np.pi * np.arange(-M, M + 1) / a
nG = len(G)
nk = 121
ks = np.linspace(-np.pi / a, np.pi / a, nk)


def band_structure(V0):
    """Eigenvalues of the plane-wave Hamiltonian at every k in the zone."""
    bands = np.zeros((nk, nG))
    for i, k in enumerate(ks):
        H = np.diag(0.5 * (k + G) ** 2).astype(float)
        for m in range(nG - 1):          # V0*cos couples G and G +/- 2pi/a
            H[m, m + 1] += V0 / 2
            H[m + 1, m] += V0 / 2
        bands[i] = np.linalg.eigvalsh(H)
    return bands


free = band_structure(0.0)
weak = band_structure(4.0)
strong = band_structure(12.0)

gap1_weak = weak[0, 1] - weak[0, 0]        # first gap sits at k = -pi/a (zone edge)
gap1_strong = strong[0, 1] - strong[0, 0]
print("First band gap at the zone boundary:")
print(f"  V0 = 4  : gap = {gap1_weak:.3f}  (nearly-free-electron theory predicts ~V0 = 4)")
print(f"  V0 = 12 : gap = {gap1_strong:.3f}")
print("\nBand filling logic (2 electrons per k-state, spin up+down):")
print("  1 electron / cell  -> band 1 HALF full        -> METAL")
print("  2 electrons / cell -> band 1 FULL, gap above  -> INSULATOR/SEMICONDUCTOR")
print("  This single counting rule explains Na (metal) vs Si (semiconductor).")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
fig.suptitle("Lesson 6 — Gaps open as the crystal potential grows (plane-wave bands)",
             fontsize=13, fontweight="bold")

for ax, bands, V0 in zip(axes, [free, weak, strong], [0, 4, 12]):
    for b in range(4):
        ax.plot(ks * a / np.pi, bands[:, b], lw=2)
    ax.set_title(f"V0 = {V0}" + ("  (free electrons)" if V0 == 0 else ""))
    ax.set_xlabel("k  (units of π/a)")
    ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("energy (hartree)")
axes[0].set_ylim(0, 60)

# shade the first gap on the strong-potential panel
axes[2].axhspan(strong[0, 0], strong[0, 1], color="#d62728", alpha=0.15)
axes[2].text(0.02, (strong[0, 0] + strong[0, 1]) / 2, "band gap",
             color="#d62728", va="center", fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = Path(__file__).parent / "lesson06_bands.png"
fig.savefig(out, dpi=150)
print(f"\nSaved figure to {out}")
plt.show()
