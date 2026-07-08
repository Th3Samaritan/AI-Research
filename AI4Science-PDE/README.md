# AI4Science — PDEs
### A practical roadmap: from "what is a PDE?" to building neural PDE solvers on real problems

> **The thesis of this track:** you cannot judge whether a neural network solved a PDE
> unless you can solve that PDE yourself the classical way. So every phase pairs
> *one classical skill* with *one real problem*, and the ML phases reuse the exact
> problems you already solved numerically — so you always have ground truth.

**Working style** (same as `Paper To Code/`): every phase ends with an artifact —
a notebook, an animated site, or a written note. No phase is "done" by reading alone.

---

## The map (7 phases, ~2 capstones)

| Phase | Theme | Real problem you solve | Core tool |
|---|---|---|---|
| 0 | PDE literacy | Coffee cup cooling on your desk | NumPy |
| 1 | Classical numerics | Lid-driven cavity flow (real CFD benchmark) | NumPy/SciPy |
| 2 | Autodiff as a science tool | Recover a hidden diffusion coefficient from noisy data | PyTorch |
| 3 | PINNs | Burgers shock + groundwater inverse problem | PyTorch / DeepXDE |
| 4 | Neural operators | Darcy flow & Navier–Stokes (FNO paper benchmarks) | `neuraloperator`, PDEBench |
| 5 | Meshes & rollouts | Flow over an airfoil / cloth & fluid sims | GNNs (MeshGraphNets) |
| 6 | Frontier | Weather forecasting on real ERA5 data | FourCastNet / GraphCast lineage |

---

## Phase 0 — PDE literacy (1–2 weeks)
*Goal: read ∂u/∂t = α∇²u aloud and know what every symbol is doing physically.*

**Learn**
- [ ] What a PDE is: a rule relating rates of change in time and space. The big four families:
  - **Diffusion/heat** `u_t = α u_xx` — smoothing (heat, chemicals, option prices)
  - **Advection** `u_t + c u_x = 0` — transport (wind carrying smoke)
  - **Wave** `u_tt = c² u_xx` — oscillation (sound, seismic)
  - **Poisson/Laplace** `∇²u = f` — equilibrium (electrostatics, steady heat)
- [ ] Initial conditions vs. boundary conditions (Dirichlet / Neumann / periodic) — and why a PDE without them is not a problem yet.
- [ ] Finite differences from Taylor series: derivative ≈ (u[i+1] − u[i−1]) / 2Δx.

**Real problem — "the coffee cup"**
- [ ] Measure (or take realistic values for) a cooling cup of coffee. Model it as 1D radial heat diffusion with a Robin/convective boundary. Solve with explicit finite differences in ~60 lines of NumPy. Compare against Newton's law of cooling and against your kitchen thermometer.
- [ ] Break it on purpose: take a timestep too large and watch it explode. You've just discovered the **CFL/stability limit** — the single most important fact in numerical PDEs.

**Deliverable:** notebook `00_coffee_cup.ipynb` with the animated temperature profile + a note on the stability limit you measured vs. the theory (Δt ≤ Δx²/2α).

**Resources:** 3Blue1Brown "But what is a PDE?" · Strang, *Computational Science and Engineering* ch. 1.

---

## Phase 1 — Classical numerics: earn your ground truth (3–5 weeks)
*Goal: solve the PDEs everyone benchmarks neural nets on — yourself, from scratch.*

**Learn**
- [ ] Explicit vs. implicit time stepping (forward Euler, backward Euler, Crank–Nicolson); why implicit buys stability at the cost of solving a linear system.
- [ ] Upwinding for advection and why naive central differences produce wiggles.
- [ ] 2D: five-point Laplacian, solving Poisson with Jacobi/Gauss–Seidel/conjugate gradient.
- [ ] Spectral methods on periodic domains (FFT → derivatives are multiplications). *This is exactly the intuition behind the Fourier Neural Operator later.*
- [ ] What FEM/FVM are and when engineers use them (read-only; don't implement).

**Real problems — the classic ladder** (follow Lorena Barba's *12 Steps to Navier–Stokes*)
- [ ] 1D advection: transport a pollutant plume down a river; watch upwinding vs. central.
- [ ] **1D Burgers' equation** `u_t + u u_x = ν u_xx`: a traffic-flow / shock-wave model. Watch a smooth wave steepen into a shock. *You will reuse this exact problem in Phase 3 (PINNs).*
- [ ] 2D Poisson: steady temperature of a CPU die with hot spots.
- [ ] **Capstone: 2D lid-driven cavity** (incompressible Navier–Stokes) — the "hello world" of real CFD. Validate your centerline velocities against the published Ghia et al. (1982) benchmark table. If your numbers match a 1982 journal paper, you are doing real computational science.
- [ ] Spectral bonus: 2D turbulence / vorticity equation with FFTs — pretty, and directly sets up FNO.

**Deliverable:** `01_twelve_steps/` notebooks + one animated site (house style) showing the cavity flow spinning up, with the Ghia validation plot.

**Resources:** Lorena Barba, *CFD Python: 12 steps to Navier–Stokes* (free) · Trefethen, *Spectral Methods in MATLAB* (skim).

---

## Phase 2 — Autodiff as a scientific instrument (1–2 weeks)
*Goal: see automatic differentiation as the bridge between simulation and learning.*

**Learn**
- [ ] `torch.autograd.grad` on *inputs*, not just weights: compute u_x, u_xx of a network u(x,t) exactly. This one trick **is** the core of PINNs.
- [ ] Optimizing through a simulation: write your Phase-0 heat stepper in PyTorch, make α a `Parameter`, and fit it by gradient descent through the whole rollout (**differentiable physics**).

**Real problem — a real inverse problem**
- [ ] Generate noisy temperature sensor readings from your coffee-cup sim with a "true" hidden α, then recover α from the data alone by backprop through the solver. Add noise until it breaks; note where.

**Deliverable:** `02_diff_physics.ipynb` — recovered α vs. truth vs. noise level curve.

---

## Phase 3 — Physics-Informed Neural Networks (3–4 weeks)
*Goal: train networks whose loss is the PDE residual itself; know when PINNs shine (inverse problems) and when they embarrass themselves (forward stiff problems).*

**Learn**
- [ ] The PINN recipe (Raissi, Perdikaris & Karniadakis 2019): loss = data misfit + PDE residual at collocation points + BC/IC terms.
- [ ] Failure modes and fixes: loss-term balancing, Fourier features / SIREN activations for high frequencies, curriculum in time, why L-BFGS after Adam.
- [ ] The honest comparison: your Phase-1 finite-difference Burgers solver runs in milliseconds; the PINN takes minutes. **PINNs earn their keep on inverse and data-fusion problems, not clean forward ones.**

**Real problems**
- [ ] **Burgers shock (the paper's own benchmark):** reproduce Raissi's Fig. 1 — and validate against *your own* Phase-1 solver, not the paper's picture.
- [ ] **Groundwater / Darcy inverse problem:** given sparse pressure-well measurements, recover the hidden permeability field. This is a real hydrology task (contaminant tracking, aquifer management) and the setting where PINNs genuinely beat classical workflows.
- [ ] Data fusion: sprinkle a few "sensor" points from the cavity flow into a PINN and reconstruct the full velocity field (flow reconstruction from sparse sensors — real wind-tunnel use case).

**Deliverable:** paper-to-code treatment of Raissi 2019 (math PDF + animated site, like `scaling law/`), plus the groundwater notebook.

**Resources:** Raissi et al. 2019 · Karniadakis et al., "Physics-informed machine learning" (Nature Reviews Physics 2021) · DeepXDE library & docs (use it *after* one from-scratch PINN).

---

## Phase 4 — Neural operators: learn the solver, not the solution (3–4 weeks)
*Goal: the conceptual jump that powers modern AI4Science — learn the **operator** mapping (coefficients/IC) → solution, so one trained model answers infinitely many problem instances in milliseconds.*

**Learn**
- [ ] Why a PINN solves *one* problem but an operator solves a *family* — the amortization argument (and its scaling-laws flavor: pay compute once at training, then inference is nearly free).
- [ ] **DeepONet** (branch/trunk nets, Lu et al. 2021) — universal approximation for operators.
- [ ] **Fourier Neural Operator** (Li et al. 2021): convolution in Fourier space = global receptive field; discretization-invariance (train 64², evaluate 256²). *Your Phase-1 spectral solver is the intuition.*
- [ ] Benchmark discipline: PDEBench / PDEArena datasets, proper train/test over parameter distributions, resolution-generalization tests.

**Real problems**
- [ ] **Darcy flow** (FNO paper benchmark): permeability field → pressure field. Train FNO with the `neuraloperator` library; beat/match the paper's relative L2. Then test on resolutions never seen in training.
- [ ] **2D Navier–Stokes vorticity** (FNO paper): predict turbulence rollouts; measure where autoregressive rollout error explodes.
- [ ] Train DeepONet and FNO on *the same* Darcy data and write a one-page honest comparison (accuracy, data-efficiency, speed vs. your classical solver).

**Deliverable:** paper-to-code treatment of the FNO paper (site + math PDF — it's a perfect candidate: the spectral-conv math animates beautifully), plus benchmark tables.

**Resources:** Li et al., *FNO* (2021) · Lu et al., *DeepONet* (2021) · Kovachki et al., *Neural Operator* (JMLR 2023) · `neuraloperator` (PyTorch) · PDEBench (NeurIPS 2022).

---

## Phase 5 — Irregular geometry: graphs, meshes, rollouts (2–3 weeks)
*Goal: real engineering lives on unstructured meshes, not square grids.*

**Learn**
- [ ] Message-passing GNNs as learned local stencils (a finite-volume method with learned fluxes).
- [ ] **MeshGraphNets** (Pfaff et al. 2021): encode–process–decode on simulation meshes; noise injection for stable long rollouts.
- [ ] Geometry-capable operators: Geo-FNO / GINO (read-only is fine).

**Real problems**
- [ ] **Flow over an airfoil / cylinder** from the MeshGraphNets datasets: train a GNN simulator, measure drag/lift rollout drift vs. ground truth.
- [ ] Alternative if compute is tight: 1D shallow-water (dam-break/tsunami toy) with a small GNN vs. your finite-volume solution.

**Deliverable:** one rollout-video notebook + a note: *"when does the learned simulator drift, and what did noise injection change?"*

---

## Phase 6 — The frontier: weather, foundations, and taste (open-ended)
*Goal: touch the systems that made AI4Science famous, on real data.*

**Learn / survey**
- [ ] The weather lineage: FourCastNet (FNO-based) → Pangu-Weather → **GraphCast** (GNN) → GenCast (diffusion, probabilistic). Why ERA5 reanalysis is the shared fuel.
- [ ] PDE foundation models & the field's scaling-law era (Poseidon, DPOT) — connects directly to your `scaling law/` work: loss vs. model/data/compute curves are now being fit for PDE models too.
- [ ] Differentiable-solver hybrids: solver-in-the-loop correction (JAX-CFD / PhiFlow) — ML corrects a cheap coarse solver instead of replacing it.

**Real problems (pick ONE as Capstone A)**
- [ ] **Weather:** download a slice of ERA5 (Copernicus CDS, free), fine-tune or run inference with an open checkpoint (FourCastNet / GraphCast via `ai-models`), and verify a 3-day forecast of a real named storm against what actually happened. ACC/RMSE vs. climatology baseline.
- [ ] **Engineering:** train an operator surrogate on an airfoil dataset (e.g., AirfRANS) and use it inside a genuine design loop — optimize a shape for lift/drag, then check the optimum with the real solver.
- [ ] **Hybrid:** coarse spectral turbulence solver + learned correction; show the hybrid at 64² matches the pure solver at 256² for a fraction of the cost.

**Capstone B (synthesis):** a "Neural PDE Solvers, Animated" hub site — Burgers/Darcy/NS problems with classical vs. PINN vs. FNO toggles, live error meters, in the house style. That artifact *is* the proof of solid understanding.

---

## Cross-cutting habits (every phase)
- **Always have ground truth.** Never evaluate a neural solver against nothing — that's the whole point of Phase 1.
- **Report three numbers, not one:** accuracy (relative L2), wall-clock speedup *including* training amortization, and out-of-distribution behavior (new resolution / new parameter range / longer rollout).
- **One paper-to-code artifact per phase** feeds the existing hub — suggested order: Raissi PINN (Phase 3), FNO (Phase 4), GraphCast (Phase 6).

## The bookshelf (keep it short)
1. Barba — *CFD Python: 12 Steps to Navier–Stokes* (free, do it all)
2. Raissi et al. 2019 — *Physics-Informed Neural Networks* (JCP)
3. Karniadakis et al. 2021 — *Physics-informed machine learning* (Nat Rev Phys — the field map)
4. Li et al. 2021 — *Fourier Neural Operator*
5. Lu et al. 2021 — *DeepONet* (Nat Mach Intell)
6. Kovachki et al. 2023 — *Neural Operator* (JMLR — the theory)
7. Pfaff et al. 2021 — *MeshGraphNets* (ICML)
8. Takamoto et al. 2022 — *PDEBench* (NeurIPS — benchmark discipline)
9. Lam et al. 2023 — *GraphCast* (Science)
10. Brandstetter et al. 2022 — *Message Passing Neural PDE Solvers* (ICLR)

## Suggested folder layout as work lands
```
AI4Science-PDE/
├── README.md                ← this roadmap (tick the boxes)
├── 00_foundations/          ← coffee cup, stability experiments
├── 01_classical/            ← 12 steps, cavity flow + Ghia validation
├── 02_diff_physics/         ← autodiff inverse problems
├── 03_pinns/                ← Raissi paper-to-code, groundwater inverse
├── 04_neural_operators/     ← FNO/DeepONet on Darcy & NS, PDEBench
├── 05_mesh_gnn/             ← MeshGraphNets airfoil
└── 06_frontier/             ← ERA5 weather capstone / hybrid solver
```

---
*Track started 2026-07-04 · companion to `Paper To Code/` · Abdulsamad Teniola Muyideen*
