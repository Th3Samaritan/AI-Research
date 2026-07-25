"""
AI4DFT Lesson 8 — Machine-Learned Interatomic Potentials (CHGNet)
=================================================================

FIRST install (this pulls PyTorch deps; takes a few minutes):
    pip install chgnet
Then run me with:
    python lesson08_chgnet_mlip.py
(First run also downloads the pretrained model weights, ~30 MB.)

THE AI4DFT IDEA IN ONE PARAGRAPH:
DFT is accurate but slow — minutes-to-hours PER structure. So researchers ran
DFT millions of times (the Materials Project database), then trained a graph
neural network to map {atomic structure} -> {energy, forces} directly.
CHGNet (Deng et al., Nature Machine Intelligence 2023) was trained on ~1.5
million DFT results spanning 89 elements. The result: DFT-level energies in
MILLISECONDS — a ~10,000x speedup — which unlocks long molecular dynamics,
massive screening, and materials discovery (this is how GNoME found 2.2
million candidate crystals). The catch: it's only as good as its training
data, and it can fail silently outside it. Trust, but verify against DFT.

WHAT WE DO:
  1. Build a rocksalt MgO crystal, RANDOMLY RATTLE the atoms (a "damaged"
     structure), and let CHGNet relax it back — watch forces -> ~0.
  2. Time single-point energy evaluations: the speed that changes the game.
Note: CHGNet uses the same ASE calculator interface as Lesson 7 — the
workflow really is the same, only the physics engine changed.
"""
import time
import numpy as np

from ase.build import bulk
from ase.optimize import BFGS
from chgnet.model import CHGNetCalculator

print("Loading pretrained CHGNet (downloads weights on first run)...")
calc = CHGNetCalculator()          # the trained GNN, wrapped as an ASE calculator

# ---------------------------------------------------------------------------
# 1) Relax a rattled MgO crystal
# ---------------------------------------------------------------------------
atoms = bulk("MgO", "rocksalt", a=4.21) * (2, 2, 2)   # 16-atom supercell
atoms.calc = calc
e_perfect = atoms.get_potential_energy()

rng = np.random.default_rng(seed=42)
atoms.positions += rng.normal(0, 0.12, atoms.positions.shape)  # damage it
e_rattled = atoms.get_potential_energy()
f_max0 = np.abs(atoms.get_forces()).max()

print(f"\nPerfect MgO supercell energy : {e_perfect:10.4f} eV")
print(f"Rattled  (damaged) energy    : {e_rattled:10.4f} eV "
      f"(+{e_rattled - e_perfect:.3f} eV of strain)")
print(f"Max force on an atom         : {f_max0:8.3f} eV/Å")
print("\nRelaxing with BFGS (the same optimizer real DFT studies use):")

opt = BFGS(atoms, logfile="-")     # prints energy & max force each step
opt.run(fmax=0.02)                 # stop when all forces < 0.02 eV/Å

e_final = atoms.get_potential_energy()
print(f"\nRelaxed energy: {e_final:.4f} eV  "
      f"(recovered {e_rattled - e_final:.3f} of the {e_rattled - e_perfect:.3f} eV strain)")

# ---------------------------------------------------------------------------
# 2) The speed that changes everything
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
n_evals = 20
for _ in range(n_evals):
    atoms.positions += 0.001       # nudge so nothing is cached
    atoms.get_potential_energy()
ms = (time.perf_counter() - t0) / n_evals * 1000

print(f"\nCHGNet single-point energy: ~{ms:.0f} ms per evaluation (on CPU!)")
print("Real DFT on this 16-atom cell: minutes per evaluation, on a cluster.")
print("That gap — DFT accuracy at classical-potential speed — is the entire")
print("reason 'AI4DFT' exists. Next question (Phase 5): when should you NOT")
print("trust it? (Hint: anything far from its Materials Project training data.)")
