# Oral-exam question bank

Questions for **spoken/written defence** — deeper than the multiple-choice items in
`AI4DFT/exam.html`. Each entry gives the question, what a 3-scoring answer contains, the
misconception to watch for, and a follow-up to use when the first answer is thin.

Levels: **R** recall · **A** apply · **D** diagnose · **T** transfer (links two modules).

---

## Module 1 — Quantum minimum

**1.1 (R)** *Why does confining a particle quantise its energy?*
- **3-answer:** the wavefunction must vanish at the walls, so only whole numbers of half
  wavelengths fit; each allowed wavelength maps to one allowed momentum and hence one energy.
  Quantisation is a boundary-condition effect, not a property of the particle.
- **Watch for:** "because energy comes in packets" — that's a restatement, not a mechanism.
- **Follow-up:** what happens to the spacing as L → ∞, and why does that make free electrons
  in a big metal look continuous?

**1.2 (A)** *You discretise −½ψ″ + Vψ = Eψ on a grid. What are the matrix, its eigenvalues and
its eigenvectors, physically?*
- **3-answer:** matrix = kinetic stencil (tridiagonal, from the second-difference formula)
  plus V on the diagonal; eigenvalues = energy levels; eigenvectors = the wavefunctions
  sampled at the grid points. Finer grid = better resolution of fast-wiggling high states.
- **Watch for:** thinking the eigenvector is a probability. It is ψ; |ψ|² is the probability.
- **Follow-up:** why is the error worse for high n than low n?

**1.3 (D)** *Your box energies are 3 % too low across the board. Grid or physics?*
- **3-answer:** grid. The finite-difference kinetic operator underestimates curvature, so
  energies come out systematically low; halving dx should cut the error ~4× (second order).
  A physics error would not scale away with dx.
- **Follow-up:** how would you *prove* it's the grid without knowing the analytic answer?
  (Refine and extrapolate.)

**1.4 (A)** *State the variational principle and explain why it makes "lower is better" safe.*
- **3-answer:** E[ψ_trial] ≥ E_exact for any normalised trial function, because expanding the
  trial state in exact eigenstates gives a weighted average of eigenvalues, all ≥ E₀. So the
  energy is a one-sided ruler: it can never go below truth, hence lower = closer.
- **Watch for:** "it means the answer is within a few percent" — no bound of that kind exists.

**1.5 (T)** *Connect the double-well level splitting to the chemical bond.*
- **3-answer:** two identical wells share a barrier; tunnelling mixes the two localised states
  into symmetric (bonding, lower) and antisymmetric (antibonding, higher) combinations. Put
  two electrons in the lower one and the system's energy drops — that energy drop *is* the
  covalent bond. Splitting shrinks roughly exponentially with separation.

**1.6 (A)** *Why do bigger basis sets always lower the energy, and what stops the improvement?*
- **3-answer:** adding functions enlarges the space you minimise over, and a minimum over a
  larger set can only be ≤; convergence stalls at the exact ground state of the *Hamiltonian
  you wrote down* — after that, remaining error is physics (the model), not basis.

---

## Module 2 — The wall

**2.1 (A)** *Do the arithmetic for iron (26 electrons) on a 10-point-per-axis grid, and say
what the number means.*
- **3-answer:** 3N = 78 dimensions → 10⁷⁸ numbers, comparable to the ~10⁸⁰ atoms in the
  observable universe. It means exact wavefunction storage is not "expensive", it is
  physically impossible — and no hardware improvement changes an exponential.

**2.2 (R)** *What is the Born–Oppenheimer approximation and when does it fail?*
- **3-answer:** nuclei ~1836× heavier ⇒ treat them as fixed while solving for electrons; this
  is what makes a potential energy surface exist. Fails where electronic states get close —
  conical intersections, non-adiabatic transitions, some photochemistry, strong electron-phonon
  coupling.
- **Watch for:** "because nuclei are classical" — the mass ratio is the argument, not classicality.

**2.3 (T)** *Why is the density a legitimate replacement for the wavefunction, and what did we
pay for the swap?*
- **3-answer:** Hohenberg–Kohn says the ground-state density fixes the external potential and
  therefore the Hamiltonian, so no information about the ground state is lost. The price: the
  functional that turns density into energy contains an unknown piece (xc), so exactness in
  principle becomes approximation in practice.

**2.4 (D)** *A colleague says "DFT scales as N³, so it's cheap." Correct them.*
- **3-answer:** N³ is the scaling of a single SCF solve with system size; the prefactor,
  k-points, cutoff, spin, and the number of SCF steps all matter, and hybrids/GW blow the
  scaling up again. Cheap relative to exponential ≠ cheap.

---

## Module 3 — Mean field & SCF

**3.1 (R)** *Draw the SCF loop from memory and name what each arrow does.*
- **3-answer:** guess n → build v_eff[n] (external + Hartree + xc) → solve the one-electron
  equations → new orbitals → new n → mix old/new → check convergence → repeat. The loop exists
  because the operator depends on its own solution.

**3.2 (A)** *What exactly is correlation energy, and why does 1.5 % of the total matter?*
- **3-answer:** E_c = E_exact − E_HF (complete basis). Small against total energy but the same
  order as bond energies and formation-energy differences, which are what decide stability. In
  materials work you never care about total energies, only differences — so "small" errors sit
  right on top of the quantity of interest.

**3.3 (D)** *Your SCF oscillates and then diverges on a big metallic slab. Diagnose.*
- **3-answer:** charge sloshing — long-wavelength density fluctuations are cheap in a metal, so
  simple mixing over-corrects. Fixes: lower mixing β, use Kerker/Broyden-type preconditioned
  mixing, add smearing, check the cell isn't pathological (huge vacuum, wrong magnetisation).
- **Watch for:** raising the cutoff as the first move — that's an accuracy knob, not stability.

**3.4 (T)** *Where does self-interaction come from, and name two symptoms you would see in
published numbers.*
- **3-answer:** the Hartree term integrates the density against itself, so each electron feels
  its own charge; approximate xc cancels it only partly. Symptoms: too-small band gaps,
  over-delocalised charge (a hole that should localise on one site smears over several),
  underestimated reaction barriers, spurious fractional charges on dissociation.

**3.5 (A)** *Why does the self-consistent helium density sit further out than the bare-nucleus one?*
- **3-answer:** each electron is repelled by the other's cloud, which screens the nucleus; the
  effective attraction is weaker than Z = 2, so the cloud relaxes outward. Same screening logic
  that orders 4s before 3d in real atoms.

---

## Module 4 — Hohenberg–Kohn & Kohn–Sham

**4.1 (R)** *State both HK theorems and say what each buys you.*
- **3-answer:** (1) the ground-state density determines v_ext up to a constant, hence H, hence
  everything — the density is sufficient information; (2) there is an energy functional E[n]
  minimised by the true density — so you can *find* it variationally. Together: legality plus
  a method. Neither tells you the functional.

**4.2 (A)** *Write v_eff and say what each term is.*
- **3-answer:** v_eff = v_ext + v_H[n] + v_xc[n]: nuclei; classical electrostatic repulsion of
  the density with itself; and everything left over — exchange (Pauli), correlation, and the
  difference between the true kinetic energy and the non-interacting T_s.
- **Follow-up:** which of the three is exactly known? (First two.)

**4.3 (D)** *In what sense are Kohn–Sham orbitals fictitious, and why do we still plot them?*
- **3-answer:** they solve a non-interacting auxiliary problem constructed to reproduce the true
  density; only n (and the total energy) is guaranteed. In practice their eigenvalues track
  band dispersions well and their shapes carry chemical meaning, so they are used as an
  interpretive tool — with the known caveat that KS gaps are not quasiparticle gaps. The one
  formal anchor: for the exact functional, ε_HOMO = −IP.

**4.4 (T)** *Why did Kohn and Sham reintroduce orbitals when the point of DFT was to avoid them?*
- **3-answer:** pure density functionals for kinetic energy (Thomas–Fermi) are too inaccurate to
  give bonding at all. Orbitals let you compute T_s exactly and quarantine only the small
  residue in E_xc. The cost is N orbitals instead of one density — still polynomial, and the
  accuracy is what made DFT usable.

**4.5 (A)** *"DFT is an approximate theory." True, false, or badly worded?*
- **3-answer:** badly worded. The framework is exact for the ground-state density and energy;
  every practical error comes from the approximate xc functional plus numerical settings.
  Separately, standard DFT is a ground-state theory — excited states need TDDFT/GW/BSE.

---

## Module 5 — Functionals

**5.1 (R)** *Climb Jacob's ladder: name the rungs, the extra ingredient, and one thing each fixes.*
- **3-answer:** LDA (n) → GGA/PBE (+∇n, better energetics and geometries) → meta-GGA/SCAN
  (+τ, better across diverse bonding) → hybrids/HSE06 (+exact exchange, respectable gaps) →
  RPA/GW/ML functionals. Cost climbs with each rung; accuracy usually but not always follows.

**5.2 (D)** *Your PBE band gap for silicon is 0.6 eV. Is the calculation wrong?*
- **3-answer:** no — that is the expected PBE answer. The KS gap misses the xc derivative
  discontinuity and PBE has self-interaction error, so gaps come out ~40–50 % low. If you need
  the gap, use HSE06 (~1.2 eV) or GW; if you need the geometry, PBE is fine.
- **Watch for:** blaming k-points or cutoff. Convergence errors don't produce a factor of two.

**5.3 (A)** *You need lattice constants to better than 1 %. LDA or PBE, and how do you bound it?*
- **3-answer:** LDA overbinds (a₀ ~1–2 % small), PBE underbinds (~1 % large); run both, and the
  experimental value usually sits between them — use the spread as an error bar. Better: use
  r²SCAN, and always compare to a measured value for a related compound first.

**5.4 (T)** *Why are hybrids so expensive in plane-wave codes but tolerable in Gaussian codes?*
- **3-answer:** exact exchange is non-local: each orbital pairs with every other occupied
  orbital. In plane waves that means many FFTs per pair per SCF step. Localised Gaussian bases
  make exchange integrals sparse and pre-computable, so quantum chemistry codes wear hybrids
  as their default while solid-state codes reserve them for targeted runs.

---

## Module 6 — Crystals & bands

**6.1 (R)** *State Bloch's theorem and explain what it does to the cost of the problem.*
- **3-answer:** ψ_nk = e^{ik·r} u_nk(r) with u lattice-periodic. An infinite crystal collapses to
  one unit cell plus a continuous label k, sampled on a finite mesh — that is why solids are
  computable at all, and why plane waves are the natural basis.

**6.2 (A)** *Where do gaps come from in a nearly-free-electron picture?*
- **3-answer:** at the zone boundary two plane waves differing by G are degenerate; the Fourier
  component V_G of the crystal potential couples them, splitting them by 2|V_G|. Standing waves
  pile charge on the ions (low) or between them (high). No V_G, no gap.

**6.3 (A)** *Predict metal or insulator: (a) Na, (b) Si, (c) a hypothetical crystal with 3
valence electrons per primitive cell.*
- **3-answer:** each band holds 2 per cell. (a) 1 e⁻ ⇒ half-filled ⇒ metal. (b) 8 valence e⁻ in
  a 2-atom cell fill 4 bands with a gap above ⇒ semiconductor. (c) odd ⇒ partially filled ⇒
  metal (band-theory prediction; Mott physics can override it, and DFT will not tell you).

**6.4 (D)** *Same cutoff, same functional, metal and insulator — why does the metal need 4× the
k-points?*
- **3-answer:** the BZ integrand jumps at the Fermi surface; the insulator's is smooth because
  bands are entirely full or empty. Hence dense meshes plus smearing for metals, and a
  smearing-width convergence check on top of the mesh check.

**6.5 (T)** *Connect the plane-wave cutoff to module 1's grid resolution.*
- **3-answer:** the same statement in reciprocal space: cutoff = the shortest wavelength your
  basis can represent = the finest real-space feature you can resolve. Too low a cutoff is
  exactly the too-coarse grid that made your box energies wrong — and both errors are
  variational (energy comes out too high as the basis is truncated).

---

## Module 7 — Production DFT

**7.1 (R)** *What does a pseudopotential replace, and what does it cost you?*
- **3-answer:** frozen core electrons and the nodal structure of valence states near the
  nucleus, so plane-wave cutoffs drop by an order of magnitude. Cost: transferability — the
  pseudopotential is built for a functional and a reference configuration, and results depend
  on the library. Anything involving core states (XPS, NMR, EFG) needs PAW or all-electron.

**7.2 (A)** *Write the convergence protocol you would run before quoting a formation energy.*
- **3-answer:** converge total-energy *differences* (say ≤ 1 meV/atom) with respect to (i)
  cutoff at fixed dense k-mesh, (ii) k-mesh at the chosen cutoff, per structure type; use
  identical cutoff, pseudopotentials, functional and comparable k-density for every structure
  in the comparison; state all of it in the write-up.

**7.3 (D)** *Two runs, same compound, cutoffs 400 and 520 eV. The energy difference is 2 eV.
What can you conclude about the formation energy?*
- **3-answer:** nothing — absolute energies are basis-dependent and meaningless alone; only
  differences computed with matched settings carry physics. Redo both at one cutoff.

**7.4 (A)** *What do you get from an E(V) curve and a Birch–Murnaghan fit, and how do you get it
wrong?*
- **3-answer:** V₀ (hence a₀) from the minimum and B₀ from the curvature. Get it wrong by
  straining too far (±20 %: the EOS form no longer describes the data), by too few points, by
  relaxing inconsistently, or by fitting across a phase transition.

**7.5 (T)** *Give the "is it converged / is it validated / is it in-distribution" answer for a
number you would put in a paper.*
- **3-answer:** converged = insensitive to numerical settings at a stated tolerance; validated =
  reproduces a known result (experiment, higher theory, or a benchmark compound) with the same
  settings; in-distribution = for any ML component, the configuration resembles training data —
  checked by spot-running DFT.

---

## Module 8 — AI4DFT

**8.1 (D)** *An MLIP trained on PBE data disagrees with experiment by 4 %. Whose fault?*
- **3-answer:** probably PBE's. A surrogate inherits its label-generator's bias; the honest
  test is MLIP vs the DFT it was trained on. If they agree, the model is fine and the theory is
  the limit — you need better reference data (r²SCAN, hybrids, experiment), not more training.

**8.2 (A)** *Name two situations where you should not trust an MLIP, and the check you would run.*
- **3-answer:** off-distribution geometries — bond breaking/transition states, extreme
  pressure or temperature, unusual stoichiometries, surfaces/defects if trained on bulk. Check:
  spot-run real DFT on a sample of frames along the trajectory (especially the extremes), use
  ensemble/uncertainty estimates where available, and verify any final structure with DFT.

**8.3 (T)** *Two distinct places ML enters DFT — name them and give an exemplar of each.*
- **3-answer:** (i) as a surrogate for the whole energy/force evaluation — MLIPs: CHGNet, MACE,
  M3GNet; (ii) as a component inside the theory — ML exchange–correlation functionals: DM21.
  A third, adjacent: generative structure proposal (GNoME, MatterGen) with DFT as the referee.

**8.4 (A)** *Quantify the pitch: 10⁵ structures, 10 min per DFT, 60 ms per MLIP call.*
- **3-answer:** ~1.9 years serial vs ~100 minutes — about 10⁴×. That is the ratio that converts
  "a PhD project" into "an afternoon", and it is why screening campaigns run ML-first and use
  DFT to verify the shortlist.

**8.5 (D)** *"Will MLIPs replace DFT?" — argue it properly.*
- **3-answer:** no, because their accuracy ceiling is the theory that labelled their training
  data, and their reliability collapses off-distribution — both failures are only detectable
  *with* DFT. What actually happens is a division of labour: ML explores, DFT verifies and
  re-labels, the model retrains (active learning). DFT's role shifts from workhorse to
  ground truth, which raises the value of every DFT calculation rather than lowering it.
