"""
AI4DFT Lesson 2 — The Hydrogen Atom (where the periodic table comes from)
=========================================================================

Run me with:   python lesson02_hydrogen_atom.py

Hydrogen is THE exactly solvable atom, so it is our calibration target.
In 3D the electron feels the Coulomb pull of the nucleus, V(r) = -Z/r.
Because the atom is spherically symmetric, the 3D problem splits into an
angular part (spherical harmonics -> the s, p, d shapes you know from
chemistry) and a 1D RADIAL equation for u(r) = r * R(r):

    -1/2 u''(r) + [ l(l+1)/(2 r^2)  -  Z/r ] u(r) = E u(r)

  l = angular momentum quantum number (0=s, 1=p, 2=d)
  l(l+1)/(2r^2) = "centrifugal barrier": pushes high-l electrons away
                  from the nucleus.

Exact energies (atomic units, hartree):  E_n = -Z^2 / (2 n^2), for ANY l < n.
That l-independence ("accidental degeneracy") is special to the pure -1/r
potential. In a real multi-electron atom the inner electrons SCREEN the
nucleus, the degeneracy breaks, and 4s fills before 3d — the entire
structure of the periodic table falls out of that effect.

Same numerical trick as Lesson 1: finite differences -> tridiagonal matrix
-> eigenvalues. Only the potential changed.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from pathlib import Path

Z = 1.0                            # nuclear charge (1 = hydrogen; try 2 with task!)
rmax, N = 60.0, 3000               # radial grid: (0, 60] bohr
r = np.linspace(rmax / N, rmax, N)
dr = r[1] - r[0]


def solve_radial(l, n_states=3):
    """Lowest eigenpairs of the radial equation for angular momentum l."""
    V = -Z / r + l * (l + 1) / (2 * r**2)
    main = 1.0 / dr**2 + V
    off = -0.5 / dr**2 * np.ones(N - 1)
    E, U = eigh_tridiagonal(main, off, select="i", select_range=(0, n_states - 1))
    U = U / np.sqrt(dr)            # normalize: integral of u^2 dr = 1
    return E, U


# Solve s, p, d channels
Es, Us = solve_radial(l=0)
Ep, Up = solve_radial(l=1)
Ed, Ud = solve_radial(l=2)

print("Hydrogen energy levels (hartree). Exact: E_n = -1/(2 n^2)")
print(" state   numeric      exact")
print(f"  1s   {Es[0]:9.5f}   {-Z**2/2:9.5f}")
print(f"  2s   {Es[1]:9.5f}   {-Z**2/8:9.5f}")
print(f"  2p   {Ep[0]:9.5f}   {-Z**2/8:9.5f}   <- same as 2s: degenerate!")
print(f"  3s   {Es[2]:9.5f}   {-Z**2/18:9.5f}")
print(f"  3p   {Ep[1]:9.5f}   {-Z**2/18:9.5f}")
print(f"  3d   {Ed[0]:9.5f}   {-Z**2/18:9.5f}   <- 3s = 3p = 3d")
print("\nNOTE the degeneracy in l — an 'accident' of the pure -1/r potential.")
print("Screening in multi-electron atoms breaks it => periodic table ordering.")

# ---------------------------------------------------------------------------
# Figure: radial wavefunctions + effective potentials
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.suptitle("Lesson 2 — Hydrogen: radial wavefunctions and the centrifugal barrier",
             fontsize=13, fontweight="bold")

for u, lab, c in [(Us[:, 0], "1s", "#1f77b4"), (Us[:, 1], "2s", "#5fa2d9"),
                  (Up[:, 0], "2p", "#ff7f0e"), (Ud[:, 0], "3d", "#2ca02c")]:
    ax1.plot(r, u, lw=2, color=c, label=lab)
ax1.set_xlim(0, 30)
ax1.set_title("radial functions u(r) = r·R(r)")
ax1.set_xlabel("r (bohr)"); ax1.set_ylabel("u(r)")
ax1.axhline(0, color="0.85", lw=0.8)
ax1.legend(frameon=False)

for l, c in [(0, "#1f77b4"), (1, "#ff7f0e"), (2, "#2ca02c")]:
    Veff = -Z / r + l * (l + 1) / (2 * r**2)
    ax2.plot(r, Veff, lw=2, color=c, label=f"l = {l} ({'spd'[l]})")
ax2.set_xlim(0, 15); ax2.set_ylim(-1.2, 0.6)
ax2.axhline(0, color="0.85", lw=0.8)
ax2.set_title("effective potential: Coulomb well + centrifugal barrier")
ax2.set_xlabel("r (bohr)"); ax2.set_ylabel("V_eff (hartree)")
ax2.legend(frameon=False)

for ax in (ax1, ax2):
    ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = Path(__file__).parent / "lesson02_hydrogen_atom.png"
fig.savefig(out, dpi=150)
print(f"\nSaved figure to {out}")
plt.show()
