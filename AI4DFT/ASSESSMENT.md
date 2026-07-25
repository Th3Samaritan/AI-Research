# AI4DFT — the mastery ladder

`CURRICULUM.md` tells you **what to do**. This file tells you **when you're allowed to move on**,
and how that gets decided. Nothing here is optional if the goal is to actually own the material
rather than to have read it.

## How a module works

Every module has the same four beats:

1. **Learn** — the course section (`index.html`), the prose (`dft_course.pdf`), and the
   equation-by-equation companion (`walkthrough.html`) for the lesson you're about to run.
2. **Predict** — write down, *before running*, what the script will print. One or two lines.
   A prediction you didn't write down doesn't count; it becomes a memory of having been right.
3. **Run & modify** — run the lesson, then do the code-surgery task. Change one thing, predict
   again, run again, explain the difference.
4. **Gate** — three instruments, all three required:

   | Instrument | Where | Pass bar |
   |---|---|---|
   | Recall & reasoning | `exam.html`, that module's questions | **≥ 80 %**, from memory |
   | Code surgery | the ★ task in `CURRICULUM.md` | prediction stated first, result explained |
   | Oral defence | `/dft-tutor` → *defend* mode | mean **≥ 2.4** on the 0–3 rubric, no zeros |

Then log it in `PROGRESS.md`. If a gate fails, the prescription is a named section to re-read —
never an immediate retake. Retaking a quiz you just failed tests short-term memory, which is
not the thing being built here.

## The eight modules

| # | Module | Course / PDF | Lessons you run | The gate: you can do this unaided |
|---|---|---|---|---|
| 1 | Quantum minimum | §01 | `lesson01`, `lesson02`, `lesson03` | Turn a 1D potential into a Hamiltonian matrix; say what its eigenvalues and eigenvectors *are*; state the variational principle and why it makes "lower is better" rigorous |
| 2 | The wall | §02 | (arithmetic only) | Derive 10³ᴺ from scratch, and state precisely what DFT trades away to escape it |
| 3 | Mean field & SCF | §02–03 | `lesson04` | Draw the SCF loop from memory; explain mixing and diagnose an oscillating cycle; define correlation energy and self-interaction |
| 4 | HK & Kohn–Sham | §03 | `lesson05` | State both HK theorems; write v_eff and name each term; say exactly what is fictitious about KS orbitals — and why we plot them anyway |
| 5 | Functionals | §04 | (reading + lesson05 variants) | Place LDA/PBE/SCAN/HSE06 on the ladder with one success and one failure each; explain the band-gap problem without saying "it's approximate" |
| 6 | Crystals & bands | §05 | `lesson06` | Predict metal vs semiconductor by electron counting; explain the gap at the zone edge; say why metals need denser k-meshes |
| 7 | Production DFT | §06 | `lesson07` | Write the convergence protocol for a paper-grade number; explain error cancellation and how to break it |
| 8 | AI4DFT | §07 | `lesson08` | Explain where an MLIP's accuracy comes from, construct a case where it lies silently, and defend a position on "will ML replace DFT?" |

## The four prove-it deliverables

Quizzes test recall. These test whether you can *produce* something. Each is a short artefact
you keep in the repo — a script plus a written paragraph. Do them in order; each one is due
after the module named.

**P1 — after module 1: the convergence experiment.**
Take `lesson01_schrodinger_1d.py`. Compute the n = 0 box error at 4 grid resolutions, tabulate,
and measure the observed order of accuracy from the slope. One paragraph: what order did you
measure, what order did you expect, and what would a *physics* error have looked like instead.

**P2 — after module 4: your own DFT code, dissected.**
Take `lesson05_kohn_sham_1d.py`. Produce three binding curves: full, exchange switched off, and
one other term of your choice removed. One figure, one paragraph per curve on what that term
was holding up. This is the deliverable that proves you understand what a DFT code *is*.

**P3 — after module 6: metal or not, decided twice.**
For one metal and one semiconductor, predict the classification by electron counting on paper,
then show it in the band structure from `lesson06_bands_planewaves.py` (or a real calculation if
you're on Phase 3). One paragraph on where a simple counting argument would mislead you.

**P4 — after module 8: the trust report.**
Relax a structure with CHGNet (`lesson08_chgnet_mlip.py`), then deliberately push it
off-distribution (large rattle, unusual composition, extreme strain) until the answer becomes
untrustworthy. Report: at what damage level did it break, how did you *know* it broke, and what
would you have run to catch it if you hadn't known the answer.

## Capstone rubric

The Section 6 capstone in `CURRICULUM.md` is marked on five things, not on whether the number is
right:

| Criterion | What full marks looks like |
|---|---|
| Question | A materials question with a decision attached — someone would act differently on the answer |
| Method | Settings stated: functional, cutoff, k-mesh, pseudopotentials, and *why* each |
| Convergence | Differences converged to a stated tolerance, shown, not asserted |
| Validation | At least one number compared to experiment or literature, with the discrepancy discussed |
| Honesty | The limits stated: what your functional systematically gets wrong, and where an ML component was out of distribution |

## Review schedule (so it stays learned)

Passed material decays. Run `/dft-tutor` → *review* on this cadence; it pulls questions from
modules by age, not by order.

| When | What |
|---|---|
| +7 days after a pass | 5 questions from that module |
| +30 days | 5 questions mixing that module with the one before it (transfer questions) |
| Before starting any new phase | one mixed 10-question round across everything passed |

## The rule that never expires

Every number you produce gets three questions: **Is it converged?** (grid, basis, k-points)
**Is it validated?** (against an exact result, experiment, or higher theory) **Is it
in-distribution?** (for anything ML). You will meet all three failure modes in this course on
purpose — that is what the modification tasks are for.
