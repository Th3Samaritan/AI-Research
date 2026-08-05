# AI4DFT — Full Curriculum & Task Sheet

**How to use this document.** Work top to bottom. For each section:
1. Read the matching chapter in `dft_course.pdf` (same section numbers) or the matching
   section of the interactive site (`index.html`).
2. If a formula feels like it was handed to you rather than derived, open
   **`derivations.html`** — notation dictionary, the hydrogen radial equation and every Gaussian
   integral derived from scratch, and the matrices worked out by hand so you can produce the
   printed numbers on paper before running anything (covers lessons 2-6 in full).
3. Before running a lesson, open its chapter in **`walkthrough.html`** (or
   `lesson_walkthrough.pdf` — same content): it unpacks every equation symbol by symbol,
   explains where every parameter value came from, and tells you exactly what output to
   expect. Predict the output before revealing it.
4. Run the listed Python file(s) yourself from the `lessons/` folder.
5. Do **every task** in the "Tasks" block before moving on. Tasks marked ★ are the
   checkpoint — if you can't do a ★ task, re-read before continuing.
6. **Sit the gate.** Each section maps to a module in `ASSESSMENT.md`, and a module is only
   finished when all three instruments pass: the module's questions in `exam.html` (≥ 80 %),
   the ★ code-surgery task with a written prediction, and the oral defence via `/dft-tutor`.
7. Tick the box here and log the result in `PROGRESS.md`.

> **The habit that makes this work:** before you run anything, write down what you expect it to
> print. One line is enough. Everything in this course is built to be predicted first and
> checked second — that's the difference between learning DFT and watching DFT.

Setup once: you already have Python 3.13 with numpy, scipy, matplotlib. Install per-section
extras only when a section says so.

---

## Section 1 — Quantum mechanics minimum

### 1.1 The Schrödinger equation as an eigenvalue problem
- [ ] **Run:** `python lesson01_schrodinger_1d.py`

**Concepts:** wavefunction, |ψ|² as probability, quantized energy levels, the
Hamiltonian matrix, why "solve the Schrödinger equation" = "diagonalize a matrix".

**Tasks after running:**
1. In the printed box-energy table, which way do the errors grow with n, and why?
   (Hint: higher states wiggle faster — think grid resolution.)
2. Double the grid points (`nx`-equivalent: the `800` in `xA`). By what factor does the
   n=0 error shrink? What does that tell you about the order of accuracy?
3. In the double well, change `(x**2 - 1)**2` to `(x**2 - 2.25)**2` (wells further apart).
   What happens to the bonding–antibonding splitting? Explain with the word "tunneling".
4. ★ In one paragraph, in your own words: why does confining a particle quantize its energy?

### 1.2 The hydrogen atom
- [ ] **Run:** `python lesson02_hydrogen_atom.py`

**Concepts:** radial equation, quantum numbers n and l, s/p/d, centrifugal barrier,
accidental degeneracy, screening → periodic-table ordering.

**Tasks after running:**
1. Set `Z = 2.0` (He⁺). Verify the 1s energy is ≈ −2.0 hartree, i.e. Z²·(−0.5). Nuclear
   charge matters *quadratically* — this is why core electrons are so tightly bound.
2. Looking at the effective-potential panel: why does a 2p electron spend less time near
   the nucleus than a 2s electron? Which is better at "screening" the nucleus?
3. ★ Explain to an imaginary classmate why 4s fills before 3d in real atoms even though
   hydrogen has them degenerate. (Two sentences, must use the word "screening".)

### 1.3 The variational principle & basis sets
- [ ] **Run:** `python lesson03_variational_basis_sets.py`

**Concepts:** E[trial] ≥ E_exact, "guess and minimize" as a rigorous method, Gaussian
basis sets, overlap matrix, generalized eigenvalue problem, basis-set convergence.

**Tasks after running:**
1. Add a 4th Gaussian to the STO-3G list (try exponent `0.05`). Does the energy go down?
   Can it EVER go below −0.5? Why not?
2. Remove the widest Gaussian (0.1688…) instead. Which part of the wavefunction gets
   worse — near the nucleus or the tail? (Reason from widths: small exponent = wide.)
3. ★ State the variational principle from memory, and explain why it makes "lower
   energy = better wavefunction" a safe rule.

**Section 1 checkpoint ★★:** you can now say what an orbital is (an eigenvector), what an
energy level is (an eigenvalue), and why bigger basis = better. That is the vocabulary of
every DFT paper.

---

## Section 2 — The many-electron problem

### 2.1 Why exact quantum mechanics dies (reading only)
Read PDF §2.1 / site section "The Wall". The wavefunction of N electrons lives in 3N
dimensions: on a laughably coarse 10-point-per-axis grid, iron's 26 electrons need
10⁷⁸ numbers — more than there are atoms in the universe. **This is the wall DFT exists
to climb over.** No code to run; the task is the arithmetic:

**Tasks:**
1. How many grid values would you need for silicon (14 electrons), same 10-point grid?
2. ★ Write the one-sentence version of "the curse of dimensionality" you'd give a manager.

### 2.2 Mean-field theory & the SCF loop (helium)
- [ ] **Run:** `python lesson04_hartree_helium_scf.py`

**Concepts:** electron–electron repulsion, mean field, Hartree potential, the
self-consistent field loop, density mixing, correlation energy.

**Tasks after running:**
1. Set `mix = 1.0` (no damping). Does the SCF still converge? Try `mix = 0.1` — count
   iterations. This tension (stability vs speed) is a real knob in every DFT code
   (`mixing_beta` in Quantum ESPRESSO).
2. From the density plot: the self-consistent cloud sits further out than the bare-nucleus
   one. Give the physical reason in one sentence.
3. The lesson got −2.857; exact is −2.904. Where did the missing energy go, physically?
   (What does "each electron sees only the *average* of the other" ignore?)
4. ★ Draw the SCF loop as a flowchart from memory: guess → build potential → solve →
   new density → mix → converged?

---

## Section 3 — DFT proper

### 3.1 Hohenberg–Kohn & Kohn–Sham (reading first, then the build)
Read PDF §3.1–3.2 / site "The Two Miracles" BEFORE running. The claims to absorb:
(1) the ground-state density n(x) — a function of just 3 coordinates — uniquely
determines *everything*; (2) a fictitious non-interacting system can reproduce the real
density if you hand it the right effective potential; (3) all remaining many-body
hardness is quarantined inside one term: exchange-correlation (xc).

### 3.2 Build your own DFT code
- [ ] **Run:** `python lesson05_kohn_sham_1d.py`   *(the most important run of the course)*

**Concepts:** Kohn–Sham equations, effective potential = external + Hartree + xc,
LDA-style exchange, total-energy assembly, double-counting corrections, binding curves.

**Tasks after running:**
1. Comment out the exchange term (set `v_x = 0` and `E_x = 0`, and the `V_x_int` line).
   Rerun the bond scan. Does the molecule still bind? How much does the equilibrium
   distance and binding energy change? You have just measured "what exchange does".
2. Change `N_ELEC = 2` to `4` and occupy 2 orbitals (change `n_orbitals=2` and sum both
   densities: `n = 2*phi[:,0]**2 + 2*phi[:,1]**2`). What happens to the bond? (Compare
   with He₂ — why don't two heliums bond?)
3. Move the SCF `mix` to 0.9 — watch what happens near small `d`. Recognize this from 2.2?
4. ★ Write down the Kohn–Sham effective potential from memory and say in words what each
   of its three terms is.

**Section 3 checkpoint ★★:** you have personally written/modified all the moving parts of
a working DFT code. Everything after this is "the same, but bigger and in 3D".

### 3.3 Functionals — Jacob's ladder (reading)
Read PDF §3.3 / site "The xc Zoo": LDA → GGA (PBE) → meta-GGA (SCAN) → hybrids (HSE06).
**Tasks:**
1. For each rung, note one thing it fixes and one thing it still gets wrong (table given
   in the PDF — reproduce it from memory afterwards).
2. ★ The famous failure: standard DFT (LDA/PBE) *underestimates band gaps* (Si: predicts
   ≈0.6 eV vs 1.17 eV real). Note why (the xc "derivative discontinuity") — you'll meet
   this in every band-structure paper you ever read.

---

## Section 4 — From atoms to materials

### 4.1 Crystals, bands, and gaps
- [ ] **Run:** `python lesson06_bands_planewaves.py`

**Concepts:** periodicity, Bloch's theorem, Brillouin zone, plane-wave basis, band
structure, band gaps, metal vs semiconductor vs insulator by electron counting.

**Tasks after running:**
1. Reduce `M` (plane waves) from 7 to 2, then 1. When do the low bands become wrong?
   You just discovered the plane-wave cutoff convergence test — mandatory in every real
   DFT study (`ecutwfc` in Quantum ESPRESSO).
2. The gap at the zone edge ≈ V0. Verify by running V0 = 8 and reading the printout.
3. ★ Using the filling rule (2 electrons per k-state), explain why sodium (1 valence
   electron) must be a metal and why diamond/Si (filled bands + gap) are not.

### 4.2 The professional workflow: ASE + equation of state
- [ ] **Install:** `pip install ase`
- [ ] **Run:** `python lesson07_ase_workflow_eos.py`

**Concepts:** ASE Atoms/Calculator design, equation of state, lattice constant, bulk
modulus, comparing calculation to experiment.

**Tasks after running:**
1. Add a third metal: `"Au"` (fcc, a0 = 4.078 Å, B = 173 GPa — EMT supports it). How
   close does it land?
2. Change the strain range from ±6 % to ±20 %. The fit gets *worse* — why? (What does a
   Birch–Murnaghan fit assume about how far you are from equilibrium?)
3. ★ In the script, mark the ONE line you would change to redo this with real DFT.

### 4.3 Real DFT setup (do when ready for Phase 3 proper)
- [ ] Install WSL + Ubuntu (`wsl --install` in an admin PowerShell — Windows can't run
      the big DFT engines natively).
- [ ] In WSL: `pip install gpaw ase` (plus `sudo apt install libxc-dev libblas-dev`), or
      build Quantum ESPRESSO.
- [ ] Rerun lesson07 with the GPAW calculator (swap the EMT line) for Si: target
      a0 ≈ 5.47 Å with PBE (expt 5.431 Å) — welcome to real DFT, ~1 % land.
- [ ] Convergence tests: energy vs `ecutwfc` and vs k-point mesh. **A DFT number without
      convergence tests is not a result** — this is the field's iron rule.

---

## Section 5 — AI4DFT

### 5.1 Machine-learned interatomic potentials
- [ ] **Install:** `pip install chgnet`   (few minutes; pulls torch extras)
- [ ] **Run:** `python lesson08_chgnet_mlip.py`

**Concepts:** GNN potentials, training on DFT databases (Materials Project), relaxation
with ML forces, the 10⁴× speedup, and the trust problem (out-of-distribution failure).

**Tasks after running:**
1. Increase the rattle from 0.12 to 0.5 Å and rerun. Does relaxation still recover the
   crystal? At what damage level does it land in a *different* (defective) minimum?
2. Swap MgO for NaCl (`bulk("NaCl", "rocksalt", a=5.64)`). CHGNet has seen it in
   training — does relaxation behave as well?
3. Time the relaxation, then estimate: how many YEARS would the same trajectory take at
   10 minutes per DFT step? Write the number down — that's the AI4DFT pitch in one figure.
4. ★ Name two situations where you should NOT trust an MLIP's answer, and what you'd do
   to check (answer: run real DFT on a few of its configurations — "spot-check DFT").

### 5.2 Where the field is (reading)
Read PDF §5 / site "AI4DFT": delta learning, ML functionals (DeepMind's DM21), universal
potentials (MACE-MP, CHGNet, M3GNet), GNoME's 2.2 M predicted crystals, and why DFT
databases (Materials Project, OQMD, AFLOW) made all of it possible.

**Task:** ★ one-paragraph position: "Will MLIPs replace DFT?" — argue using the
training-data dependence you saw in 5.1.

---

## Section 6 — Capstone (pick one)

- [ ] **A. Alloy stability:** use CHGNet to compute formation energies of 3–4 Cu–Au
      orderings, build a mini convex hull, compare to Materials Project values.
- [ ] **B. Battery flavor:** relax layered LiCoO₂ with and without Li removed; estimate
      the average voltage from the energy difference; compare to the ~4 V reality.
- [ ] **C. Your call:** any materials question from your B.Eng world, answered with the
      Section 4–5 toolkit, written up in 2 pages with a convergence/trust check.

---

## The rule that never expires
Every number you produce gets three questions: **Is it converged?** (grid, basis, k-points)
**Is it validated?** (vs exact result, experiment, or higher theory) **Is it in-distribution?**
(for anything ML). You saw all three fail modes in this course — that's the point.
