"""
AI4DFT Lesson 4 — Helium & the Self-Consistent Field (SCF) Loop
===============================================================

Run me with:   python lesson04_hartree_helium_scf.py

Helium: 2 electrons + a Z=2 nucleus. Now electrons repel EACH OTHER, and
we hit the central chicken-and-egg problem of electronic structure:

  * each electron's orbital depends on the potential it feels,
  * but part of that potential comes from the OTHER electron's density,
  * which depends on ITS orbital... which depends on the potential...

The fix is the SELF-CONSISTENT FIELD (SCF) loop — the beating heart of
every Hartree-Fock and DFT code ever written:

  1. guess a density n(r)
  2. build the potential:  V(r) = -Z/r + V_H[n](r)
     (V_H = the "Hartree" electrostatic potential of the electron cloud)
  3. solve the Schrödinger equation in that potential -> new orbital
  4. new orbital -> new density. Mix it gently with the old one.
  5. repeat until nothing changes: the field is SELF-CONSISTENT.

Both electrons (opposite spins) occupy the same 1s orbital. Each feels the
nucleus plus the cloud of the OTHER electron:

    -1/2 u'' + [ -Z/r + V_H(r) ] u = eps * u,
    V_H(r) = (1/r) * int_0^r n dr'  +  int_r^inf (n/r') dr'   (radial trick)

Total energy (avoid double-counting the repulsion, which both eps's contain):
    E_total = 2*eps - E_ee,   E_ee = int n(r) V_H(r) dr

Scoreboard for helium's ground-state energy (hartree):
    naive (ignore repulsion, 2 x -Z^2/2)  = -4.000   (terrible)
    this lesson (mean-field / Hartree-Fock) ~ -2.862
    exact (experiment)                     = -2.904
The missing 0.042 hartree (~1.1 eV!) is the famous CORRELATION ENERGY —
the error made by treating each electron as moving in the AVERAGE cloud of
the other. Capturing it cheaply is exactly what DFT is for. Remember 2.862.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from pathlib import Path

Z = 2.0
rmax, N = 20.0, 4000
r = np.linspace(rmax / N, rmax, N)
dr = r[1] - r[0]
off = -0.5 / dr**2 * np.ones(N - 1)


def lowest_state(V):
    """Ground state of -1/2 u'' + V u = eps u (tridiagonal, lowest eigenpair)."""
    eps, u = eigh_tridiagonal(1.0 / dr**2 + V, off, select="i", select_range=(0, 0))
    u = u[:, 0] / np.sqrt(dr)
    return eps[0], u


def hartree_potential(n):
    """V_H(r) of a radial density n(r) (one electron's worth, int n dr = 1)."""
    inner = np.cumsum(n) * dr                       # charge inside radius r
    outer = np.cumsum((n / r)[::-1])[::-1] * dr     # contribution from outside
    return inner / r + outer - n[-1] * 0.0          # both terms, standard trick


# --- SCF loop ---------------------------------------------------------------
eps, u = lowest_state(-Z / r)          # step 1: start from the bare-nucleus 1s
n = u**2
mix = 0.5                              # gentle mixing: crucial for convergence
print("SCF loop (watch it converge):")
print(" iter   eps (hartree)   E_total     change in density")
history = []
for it in range(1, 41):
    V_H = hartree_potential(n)         # field of the OTHER electron
    eps, u = lowest_state(-Z / r + V_H)
    n_new = u**2
    dn = np.trapezoid(np.abs(n_new - n), r)
    E_ee = np.trapezoid(n_new * V_H, r)
    E_tot = 2 * eps - E_ee
    history.append((it, eps, E_tot, dn))
    print(f"  {it:3d}    {eps:9.5f}    {E_tot:9.5f}     {dn:.2e}")
    n = (1 - mix) * n + mix * n_new    # step 4: mix old and new density
    if dn < 1e-8:
        print("  CONVERGED: the field is self-consistent.")
        break

print(f"\nFinal total energy: {E_tot:.4f} hartree")
print("  naive (no repulsion): -4.000")
print("  this (mean field):    %.4f" % E_tot)
print("  exact (experiment):   -2.9037")
print(f"  ==> correlation energy missed: {abs(-2.9037 - E_tot):.4f} hartree (~{abs(-2.9037-E_tot)*27.211:.2f} eV)")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
hist = np.array(history)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.suptitle("Lesson 4 — Helium: the self-consistent field loop",
             fontsize=13, fontweight="bold")

ax1.semilogy(hist[:, 0], hist[:, 3], "o-", lw=2, color="#1f77b4")
ax1.set_title("SCF convergence: density change per iteration")
ax1.set_xlabel("iteration"); ax1.set_ylabel("∫|Δn| dr  (log scale)")

u_bare = lowest_state(-Z / r)[1]
ax2.plot(r, u_bare**2, lw=2, ls="--", color="#9ecae1",
         label="bare nucleus (no e-e repulsion)")
ax2.plot(r, n, lw=2, color="#1f77b4", label="self-consistent (screened)")
ax2.set_xlim(0, 5)
ax2.set_title("electron density: repulsion pushes the cloud outward")
ax2.set_xlabel("r (bohr)"); ax2.set_ylabel("n(r)")
ax2.legend(frameon=False)

for ax in (ax1, ax2):
    ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = Path(__file__).parent / "lesson04_helium_scf.png"
fig.savefig(out, dpi=150)
print(f"\nSaved figure to {out}")
plt.show()
