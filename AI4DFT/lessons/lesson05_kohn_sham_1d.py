"""
AI4DFT Lesson 5 — Build a Toy Kohn-Sham DFT Code From Scratch
=============================================================

Run me with:   python lesson05_kohn_sham_1d.py

THIS is the lesson where DFT actually happens. Recall the two big ideas:

  * HOHENBERG-KOHN: the ground-state DENSITY n(x) uniquely determines
    everything — you don't need the monstrous many-electron wavefunction.
  * KOHN-SHAM: invent a FICTITIOUS system of non-interacting electrons,
    engineered to reproduce the REAL density. Those fake electrons obey a
    Schrödinger equation with an effective potential:

        v_eff(x) = v_ext(x) + v_Hartree[n](x) + v_xc[n](x)

    v_ext     : the nuclei's pull (known)
    v_Hartree : classical repulsion of the electron cloud (computable)
    v_xc      : EXCHANGE-CORRELATION — the rug under which ALL remaining
                quantum many-body effects are swept. Approximating v_xc
                well is the entire art of DFT.

Because v_eff depends on the density it produces, we need... the SCF loop
from Lesson 4. DFT = (Kohn-Sham equations) + (SCF) + (an xc approximation).

OUR TOY: a 1D "H2-like molecule". Two +1 nuclei, two electrons sharing one
doubly-occupied orbital. We use a softened Coulomb 1/sqrt(x^2+1) (standard
1D trick to avoid the singularity) and a simple LDA-style exchange
v_x = -(3 n / pi)^(1/3). (Honest caveat: that formula is borrowed from 3D —
in this toy it plays the ROLE of an xc functional so you see the machinery;
its numbers are illustrative, not publishable.)

PAYOFF: we scan the distance d between the nuclei and compute total energy
E(d). A MINIMUM appears — the code predicts a chemical bond and an
equilibrium bond length, from nothing but the density. That is DFT's whole
job: feed it nuclear positions, get energies -> geometry, stability, forces.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from pathlib import Path

# --- grid and interaction kernel -------------------------------------------
L, N = 16.0, 400
x = np.linspace(-L / 2, L / 2, N)
dx = x[1] - x[0]
off = -0.5 / dx**2 * np.ones(N - 1)
K = 1.0 / np.sqrt((x[:, None] - x[None, :]) ** 2 + 1.0)   # soft Coulomb kernel

N_ELEC = 2      # two electrons -> one doubly occupied Kohn-Sham orbital


def v_external(d):
    """Two soft nuclei of charge +1 at +/- d/2."""
    return (-1.0 / np.sqrt((x - d / 2) ** 2 + 1.0)
            - 1.0 / np.sqrt((x + d / 2) ** 2 + 1.0))


def solve_ks(v_eff, n_orbitals=1):
    """Solve the Kohn-Sham equation in a given effective potential."""
    eps, phi = eigh_tridiagonal(1.0 / dx**2 + v_eff, off,
                                select="i", select_range=(0, n_orbitals - 1))
    return eps, phi / np.sqrt(dx)


def scf(d, mix=0.4, tol=1e-8, max_iter=200, verbose=False):
    """Full Kohn-Sham SCF at nuclear separation d. Returns (E_total, n, eps)."""
    v_ext = v_external(d)
    eps, phi = solve_ks(v_ext)                 # initial guess: no interaction
    n = N_ELEC * phi[:, 0] ** 2
    for it in range(max_iter):
        v_H = K @ n * dx                       # Hartree potential
        v_x = -np.cbrt(3.0 * n / np.pi)        # toy LDA exchange potential
        eps, phi = solve_ks(v_ext + v_H + v_x)
        n_new = N_ELEC * phi[:, 0] ** 2
        dn = np.sum(np.abs(n_new - n)) * dx
        n = (1 - mix) * n + mix * n_new
        if verbose:
            print(f"    iter {it+1:3d}  eps = {eps[0]:9.5f}   |dn| = {dn:.2e}")
        if dn < tol:
            break
    # total energy: sum of eigenvalues minus double-counting corrections
    v_H = K @ n * dx
    E_H = 0.5 * np.sum(n * v_H) * dx                       # Hartree energy
    E_x = -0.75 * (3.0 / np.pi) ** (1 / 3) * np.sum(n ** (4 / 3)) * dx
    V_x_int = np.sum(n * -np.cbrt(3.0 * n / np.pi)) * dx   # int n*v_x
    E_nn = 1.0 / np.sqrt(d**2 + 1.0)                       # nuclear repulsion
    E_tot = N_ELEC * eps[0] - E_H - V_x_int + E_x + E_nn
    return E_tot, n, eps[0]


# ---------------------------------------------------------------------------
# Part A — one full SCF, narrated, at d = 2 bohr
# ---------------------------------------------------------------------------
print("Part A — Kohn-Sham SCF at nuclear separation d = 2.0 (watch it settle):")
E2, n2, eps2 = scf(2.0, verbose=True)
print(f"  converged: E_total = {E2:.5f} hartree")

# ---------------------------------------------------------------------------
# Part B — the payoff: scan d, find the bond
# ---------------------------------------------------------------------------
print("\nPart B — scanning bond length d ...")
ds = np.linspace(1.0, 8.0, 29)
Es = np.array([scf(d)[0] for d in ds])
i_eq = np.argmin(Es)
print(f"  equilibrium bond length d_eq = {ds[i_eq]:.2f} bohr")
print(f"  binding energy vs d=8:  {Es[-1] - Es[i_eq]:.4f} hartree")
print("  ==> The code PREDICTED a chemical bond. This E(d) curve is exactly")
print("      what real DFT produces for geometry optimization and forces.")

_, n_eq, _ = scf(ds[i_eq])
_, n_far, _ = scf(8.0)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.suptitle("Lesson 5 — A chemical bond emerges from a toy Kohn-Sham DFT code",
             fontsize=13, fontweight="bold")

ax1.plot(ds, Es, "o-", lw=2, color="#1f77b4")
ax1.plot(ds[i_eq], Es[i_eq], "*", ms=16, color="#d62728",
         label=f"equilibrium d = {ds[i_eq]:.2f}")
ax1.set_title("A — total energy vs nuclear separation")
ax1.set_xlabel("separation d (bohr)"); ax1.set_ylabel("E_total (hartree)")
ax1.legend(frameon=False)

ax2.plot(x, n_eq, lw=2, color="#1f77b4", label=f"density at d = {ds[i_eq]:.2f} (bonded)")
ax2.plot(x, n_far, lw=2, ls="--", color="#ff7f0e", label="density at d = 8 (separated)")
v = v_external(ds[i_eq])
ax2.plot(x, 0.15 * (v - v.min()), color="0.6", lw=1,
         label="external potential (scaled)")
ax2.set_xlim(-8, 8)
ax2.set_title("B — charge piles up BETWEEN the nuclei: that's the bond")
ax2.set_xlabel("x (bohr)"); ax2.set_ylabel("n(x)")
ax2.legend(frameon=False, fontsize=9)

for ax in (ax1, ax2):
    ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = Path(__file__).parent / "lesson05_kohn_sham_1d.png"
fig.savefig(out, dpi=150)
print(f"\nSaved figure to {out}")
plt.show()
