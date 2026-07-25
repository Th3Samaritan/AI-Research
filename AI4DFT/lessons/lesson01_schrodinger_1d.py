"""
AI4DFT Lesson 1 — The Schrödinger Equation as an Eigenvalue Problem
===================================================================

Run me with:   python lesson01_schrodinger_1d.py

The one equation under all of chemistry and materials science (1D, time-independent,
atomic units where hbar = m_e = 1):

    -1/2 * psi''(x) + V(x) * psi(x) = E * psi(x)

  psi(x) = the electron's wavefunction; |psi(x)|^2 is the probability of finding
           the electron at x. In DFT language: the DENSITY n(x) = |psi(x)|^2.
  V(x)   = the potential the electron feels (the "landscape" it lives in)
  E      = an ENERGY LEVEL. Only special discrete values of E admit a valid
           solution — this is where quantization comes from.

THE KEY IDEA OF THIS LESSON:
On a grid, that equation becomes a plain matrix eigenvalue problem

    H @ psi = E * psi

where H (the "Hamiltonian") is just a matrix built from finite differences:
the same -psi'' stencil you met in the PDE track's heat equation, plus V on the
diagonal. Eigenvalues = allowed energies. Eigenvectors = orbitals. Every DFT
code on earth — VASP, Quantum ESPRESSO, all of them — is at heart doing THIS,
just in 3D with thousands of basis functions.

We solve three classic potentials:
  A) Infinite square well ("particle in a box")  — has exact answers to check against
  B) Harmonic oscillator                          — evenly spaced levels (phonons!)
  C) Double well                                  — a first look at BONDING
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from pathlib import Path


def solve_schrodinger(x, V):
    """Solve -1/2 psi'' + V psi = E psi on grid x. Returns (energies, states).

    The second derivative on a grid is the familiar 3-point stencil:
        psi''(x_i) ~ (psi[i+1] - 2 psi[i] + psi[i-1]) / dx^2
    so H is TRIDIAGONAL: main diagonal 1/dx^2 + V, off-diagonals -1/(2 dx^2).
    Grid endpoints act as infinite walls (psi = 0 there).
    """
    dx = x[1] - x[0]
    main = 1.0 / dx**2 + V          # diagonal of H
    off = -0.5 / dx**2 * np.ones(len(x) - 1)   # off-diagonal
    energies, states = eigh_tridiagonal(main, off)
    # Normalize each state so that integral |psi|^2 dx = 1 (total probability 1)
    states = states / np.sqrt(dx)
    return energies, states


def plot_levels(ax, x, V, energies, states, n_show, scale, title):
    """Textbook-style plot: potential + wavefunctions drawn at their energy heights."""
    ax.plot(x, V, color="0.35", lw=2, label="potential V(x)")
    for n in range(n_show):
        ax.axhline(energies[n], color="0.8", lw=0.7, xmin=0.05, xmax=0.95)
        ax.plot(x, energies[n] + scale * states[:, n], lw=1.8,
                label=f"n={n}, E={energies[n]:.3f}")
    ax.set_title(title)
    ax.set_xlabel("x (bohr)")
    ax.set_ylabel("energy (hartree)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# A) Particle in a box: V = 0 inside [0, 1], infinite walls at the edges.
#    Exact answer: E_n = (n+1)^2 * pi^2 / 2   (n = 0, 1, 2, ...)
# ---------------------------------------------------------------------------
xA = np.linspace(0, 1, 800)
VA = np.zeros_like(xA)
EA, SA = solve_schrodinger(xA, VA)

print("A) Particle in a box — numeric vs exact energies (hartree)")
print("   n   numeric      exact       error")
for n in range(4):
    exact = (n + 1) ** 2 * np.pi**2 / 2
    print(f"   {n}   {EA[n]:9.4f}   {exact:9.4f}   {abs(EA[n]-exact):.2e}")

# ---------------------------------------------------------------------------
# B) Harmonic oscillator: V = 1/2 x^2. Exact: E_n = n + 1/2, EVENLY spaced.
#    This is the quantum spring — atoms vibrating in a crystal (phonons).
# ---------------------------------------------------------------------------
xB = np.linspace(-8, 8, 1200)
VB = 0.5 * xB**2
EB, SB = solve_schrodinger(xB, VB)

print("\nB) Harmonic oscillator — numeric vs exact (E_n = n + 1/2)")
print("   n   numeric      exact       error")
for n in range(4):
    print(f"   {n}   {EB[n]:9.4f}   {n + 0.5:9.4f}   {abs(EB[n]-(n+0.5)):.2e}")

# ---------------------------------------------------------------------------
# C) Double well: V = 10 * (x^2 - 1)^2 — two "atoms" side by side.
#    Watch the two lowest states: an EVEN combination (bonding) and an ODD
#    one (antibonding), split by a tiny tunneling energy. This splitting IS
#    the quantum origin of the chemical bond — and, repeated over 10^23
#    atoms, of energy BANDS in solids.
# ---------------------------------------------------------------------------
xC = np.linspace(-3, 3, 1200)
VC = 10.0 * (xC**2 - 1) ** 2
EC, SC = solve_schrodinger(xC, VC)

print("\nC) Double well — the two lowest states:")
print(f"   E0 = {EC[0]:.5f}  (even  = 'bonding')")
print(f"   E1 = {EC[1]:.5f}  (odd   = 'antibonding')")
print(f"   splitting E1 - E0 = {EC[1]-EC[0]:.5f} hartree  <-- tunneling!")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.8))
fig.suptitle("Lesson 1 — Quantized energy levels from one matrix eigenvalue problem",
             fontsize=13, fontweight="bold")

plot_levels(axA, xA, VA, EA, SA, n_show=3, scale=2.0,
            title="A — Particle in a box")
axA.set_ylim(-2, 55)

plot_levels(axB, xB, VB, EB, SB, n_show=4, scale=0.6,
            title="B — Harmonic oscillator (evenly spaced!)")
axB.set_xlim(-5, 5); axB.set_ylim(0, 5)

plot_levels(axC, xC, VC, EC, SC, n_show=2, scale=1.2,
            title="C — Double well: bonding vs antibonding")
axC.set_xlim(-2, 2); axC.set_ylim(0, 12)

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = Path(__file__).parent / "lesson01_schrodinger_1d.png"
fig.savefig(out, dpi=150)
print(f"\nSaved figure to {out}")
plt.show()
