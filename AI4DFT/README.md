# AI4DFT — Density Functional Theory, from first principles to AI

A learn-by-building track for a materials engineer: what DFT actually is, why it works,
how to run real calculations, what it can compute for materials, and how AI is
reshaping the whole field.

**Ground rule:** every lesson is a script *you* run yourself. The lesson text explains
the physics first; the script makes it concrete.

## The big picture (why a materials engineer should care)

Almost every property you studied — elastic moduli, band gaps, formation energies of
phases, diffusion barriers, surface energies — is ultimately set by how *electrons*
arrange themselves around nuclei. The exact quantum equation for many electrons is
hopelessly expensive to solve. **DFT is the trick that makes it tractable**: instead of
tracking every electron's wavefunction, it works with the electron *density* n(x) — one
function of 3 coordinates instead of 3N. It is the single most-used method in
computational materials science, and the data engine behind the Materials Project,
GNoME, and modern ML interatomic potentials.

## Layout — course materials

**Learn**
- **`CURRICULUM.md`** — START HERE. Every file to run, in order, with tasks and ★ checkpoints after each section
- **`index.html`** — interactive site: live widgets for quantization, the curse of dimensionality, SCF mixing, band-gap opening, and the MLIP speedup
- **`walkthrough.html`** + **`lesson_walkthrough.pdf`** — the deep lesson-by-lesson walkthrough: every equation symbol by symbol, every parameter's value and why it was chosen, expected output, figure guides (covers lessons 1–8 + PDE Lesson 1)
- **`dft_course.pdf`** — the full prose explanation (chapters mirror the curriculum sections)
- **`lessons/`** — 8 runnable scripts, lesson01–lesson08 (pure numpy/scipy until lesson07; `pip install ase` then `pip install chgnet` when the curriculum says so)

**Be tested** — the material only counts once it survives all three
- **`exam.html`** — 42-question self-test, graded in the browser, with a full explanation after
  every answer and a best-score memory per module
- **`ASSESSMENT.md`** — the mastery ladder: 8 modules, what each gate demands, the four prove-it
  deliverables, the capstone rubric, and the review schedule
- **`/dft-tutor`** — the tutor/examiner skill (`.claude/skills/dft-tutor/`). Run it in Claude Code
  to be taught a module, drilled, orally examined against a rubric, or to have your lesson output marked
- **`PROGRESS.md`** — your ledger; the tutor reads it at the start of a session and writes to it at the end
- `README.md` — this roadmap

## Roadmap

### Phase 0 — Quantum mechanics minimum (the language of DFT)
- [ ] **Lesson 1 — The Schrödinger equation as an eigenvalue problem**: particle in a box,
      harmonic oscillator, double well (a first look at bonding); numeric vs analytic energies
- [ ] Lesson 2 — The hydrogen atom & atomic orbitals (s, p shapes; where the periodic table comes from)
- [ ] Lesson 3 — The variational principle (why guessing wavefunctions works — the engine under all of quantum chemistry)

### Phase 1 — The many-electron problem (why DFT had to be invented)
- [ ] Lesson 4 — Born–Oppenheimer; the curse of dimensionality (why exact QM dies at ~2 atoms)
- [ ] Lesson 5 — Hartree & Hartree–Fock: mean-field thinking, exchange, and what "correlation" means

### Phase 2 — DFT proper (the core of this track)
- [ ] Lesson 6 — Hohenberg–Kohn theorems: the density is enough (the miracle, stated honestly)
- [ ] Lesson 7 — Kohn–Sham equations: fake non-interacting electrons, real density
- [ ] Lesson 8 — **Build a toy 1D DFT code from scratch**: the self-consistent field (SCF) loop
- [ ] Lesson 9 — Exchange-correlation functionals: LDA → GGA (PBE) → hybrids; "Jacob's ladder" and what accuracy to expect

### Phase 3 — Practical DFT on real materials
- [ ] Plane waves, pseudopotentials, k-point sampling, convergence testing
- [ ] Set up a real engine (Quantum ESPRESSO or GPAW via WSL — Windows needs WSL for these) + ASE as the Python driver
- [ ] First real calculation: lattice constant & bulk modulus of Si and Al (equation of state)

### Phase 4 — Materials applications
- [ ] Band structures & density of states (metal vs semiconductor vs insulator)
- [ ] Formation energies & convex hulls (phase stability — alloy design)
- [ ] Elastic constants; surfaces & adsorption; point defects
- [ ] Phonons & thermal properties

### Phase 5 — AI4DFT
- [ ] Machine-learned interatomic potentials: train/use MACE / CHGNet / M3GNet (DFT accuracy at 1000× speed)
- [ ] Delta-learning & surrogate models; ML exchange-correlation functionals (DM21)
- [ ] Materials discovery at scale: Materials Project API, GNoME-style screening

### Phase 6 — Capstone
- [ ] Pick a materials question (e.g., alloy phase stability or battery cathode screening),
      answer it with DFT + an MLIP, validated against published data

## Progress log

| Date | Lesson | Status |
|------|--------|--------|
| 2026-07-21 | Full course package built (curriculum, PDF, site, lessons 1–8) | ready |
| 2026-07-21 | Lesson 1 — Schrödinger as eigenvalue problem | ready for user to run |
| 2026-07-25 | Assessment system added (exam.html, ASSESSMENT.md, PROGRESS.md, /dft-tutor) | ready |
| 2026-07-25 | Site redesigned — light theme with dark mode, shared `assets/` design system | done |

Per-module status lives in **`PROGRESS.md`**, not here.
