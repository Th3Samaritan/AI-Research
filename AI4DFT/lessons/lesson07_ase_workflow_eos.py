"""
AI4DFT Lesson 7 — The Real Workflow: ASE, Equations of State, Bulk Modulus
==========================================================================

FIRST install ASE:      pip install ase
Then run me with:       python lesson07_ase_workflow_eos.py

ASE (Atomic Simulation Environment) is the Python framework the community
uses to BUILD structures, DRIVE calculators, and ANALYZE results. The key
design idea is the CALCULATOR interface: your script stays identical whether
energies come from:
    * EMT      — a fast classical potential (used here; ships with ASE, runs anywhere)
    * GPAW / Quantum ESPRESSO / VASP — real DFT (Phase 3, via WSL)
    * CHGNet / MACE — machine-learned potentials (Lesson 8)
Learn the workflow once with a cheap calculator; swap in DFT later by
changing ONE line. This is exactly how professionals prototype.

THE PHYSICS TASK — a classic first calculation, and a materials-engineering
staple: squeeze and stretch a crystal, record E(V), fit an EQUATION OF
STATE. The fit yields:
    * V0 -> equilibrium lattice constant a0   (compare to X-ray diffraction!)
    * B  -> bulk modulus                       (stiffness under pressure)
This E(V) -> (a0, B) pipeline is IDENTICAL in real DFT papers.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState
from ase.units import GPa

# Experimental reference values (room temperature)
EXPT = {"Al": {"a0": 4.05, "B": 76.0}, "Cu": {"a0": 3.615, "B": 140.0}}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
fig.suptitle("Lesson 7 — Equation of state with ASE: lattice constant & bulk modulus",
             fontsize=13, fontweight="bold")

print(f"{'metal':>6} {'a0 calc (Å)':>12} {'a0 expt':>9} {'B calc (GPa)':>13} {'B expt':>8}")
for metal, ax in zip(["Al", "Cu"], axes):
    volumes, energies = [], []
    for scale in np.linspace(0.94, 1.06, 9):          # squeeze/stretch +/- 6 %
        atoms = bulk(metal, "fcc", a=EXPT[metal]["a0"] * scale)
        atoms.calc = EMT()                            # <-- swap for DFT later!
        volumes.append(atoms.get_volume())
        energies.append(atoms.get_potential_energy())

    eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
    v0, e0, B = eos.fit()
    a0 = (4 * v0) ** (1 / 3)                          # fcc: 4 atoms' volume per cube...
    # NOTE: ase.build.bulk('fcc') gives the 1-atom primitive cell, V = a^3/4
    print(f"{metal:>6} {a0:12.3f} {EXPT[metal]['a0']:9.3f} "
          f"{B / GPa:13.1f} {EXPT[metal]['B']:8.1f}")

    vfit = np.linspace(min(volumes), max(volumes), 200)
    ax.plot(volumes, energies, "o", color="#1f77b4", label="calculated points")
    eos.plot(ax=ax)                                    # ASE's own EOS-fit plot
    ax.set_title(f"{metal} (fcc) — Birch-Murnaghan fit")

print("\nEMT is a cheap 1990s classical potential — decent for Cu, rougher for Al.")
print("In Phase 3 you rerun THIS EXACT SCRIPT with a DFT calculator and watch")
print("the numbers land within ~1% of experiment. The workflow won't change.")

fig.tight_layout(rect=[0, 0, 1, 0.94])
out = Path(__file__).parent / "lesson07_eos.png"
fig.savefig(out, dpi=150)
print(f"\nSaved figure to {out}")
plt.show()
