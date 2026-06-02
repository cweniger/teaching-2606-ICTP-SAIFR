# Hands-on Plan

Two 2-hour hands-on sessions, sandwiched between the lectures.

- **Session 1:** after Lec 2. Students have classic ABC (Lec 1b) + the
  Gaussian-head q_φ(θ|x) toolkit (Lec 2). No flows yet, no summary
  networks yet, no diagnostics yet.
- **Session 2:** after Lec 5. Students have the full school: flows, FM,
  diffusion, summary networks, diagnostics, sequential methods.

The two sessions form an arc: session 1 lets students *feel* why neural
SBI exists; session 2 lets them *build* a research-quality SBI pipeline
end-to-end on a problem they choose.

---

## Session 1 (2h, after Lec 2): PyTorch crash + Gaussian-head NPE

### Premise

Most students will not have written a training loop. Without ~30 min of
hands-on PyTorch, everything in Lec 2 stays magic. After that primer, do
**one method in full depth** — Gaussian-head NPE — on the ball-throw
toy, then let students pick one APP-flavoured example to apply it to.

Skip classical ABC implementation. Lec 1b's animated demos already teach
what ABC is; re-coding rejection ABC by hand is a 20-line exercise that
adds no insight and costs 30 min.

### Three-block structure (~120 min)

#### Block 1 — PyTorch crash (~30 min)

Goal: students have run a training loop *themselves* before any SBI
appears.

- Tensors, broadcasting, GPU vs CPU
- `autograd` in one line: `loss.backward()`
- `nn.Module` + `forward`
- `torch.optim` + the canonical training loop
- End with one tiny working regression example (MLP fits sin(x)) so
  they've watched the whole loop run on real data.

#### Block 2 — Gaussian-head NPE on the ball-throw (~45 min)

Goal: build the method from scratch using the PyTorch primitives just
learned. Hand-built summary statistic. No neural summary here — that
appears in Block 3 where it earns its keep.

- **Simulator:** ball-throw from Lec 1b (1D parameter θ = launch angle,
  1D observation x = landing position). Provided.
- **Hand-built summary:** mean landing position over n_balls (already
  used in Lec 1b's Demo 2). Optional toggle: raw single observation vs
  summary.
- **Model:** small MLP outputting (μ(x), log σ²(x)). Students fill in
  the network head and the Gaussian NLL loss.
- **Training:** train on (θ_i, x_i) pairs from the simulator. Plot
  training + validation loss curves (concept of held-out validation
  introduced).
- **Posterior:** predict q_φ(θ | x_obs); plot against analytic /
  fine-grid truth.
- **Prior dependence sub-exercise (~10 min):** retrain with a narrower
  prior, watch the posterior shift toward the new prior in low-data
  regions. *Critical lesson rarely taught explicitly:* NPE's posterior
  is prior-conditional in a way ABC's is not.

#### Block 3 — APP example, student choice (~45 min)

Three pre-defined, fully-implemented APP-flavoured simulators. Students
pick *one* and apply the same Gaussian-head NPE machinery they just
built. Neural summary appears here where it earns its keep (especially
for the image case).

- **Option A — GW chirp parameter inference.** Time-series observation.
  Two parameters (e.g., chirp mass + luminosity distance). Hand-built or
  small-1D-CNN summary. Easiest to author (canonical waveform models
  available in `pycbc` / `bilby`).
- **Option B — Cosmic-ray spectrum analysis.** 1D histogram observation
  (binned photon/event counts). Two parameters (e.g., spectral index +
  cutoff energy). Poisson noise. Hand-built summary or shallow MLP. CR
  problem design needs more input from CW; spec to be finalised.
- **Option C — Point-source population in images.** 2D image observation.
  Two parameters (e.g., source count + flux scale). Poisson + PSF
  forward model. Small CNN summary network. Borrow from existing
  swyft/peregrine or BayesFlow examples.

Students continue with their chosen example into Session 2 (see below).

### Decisions locked in

- **PyTorch crash is non-negotiable.** Without it, the rest is
  copy-paste-prayer.
- **One method in depth, not two methods shallow.** Gaussian-head NPE
  with proper validation and prior-dependence.
- **No flows, no diagnostics, no sequential, no NRE/NLE in Session 1.**
  All deferred to Session 2.
- **No classical ABC implementation.** Lec 1b carries it.
- **Three APP examples, student-choice.** Self-selection gives
  ownership; the variety shows the method generalising across data
  modalities.

---

## Session 2 (2h, after Lec 5): Upgrade your pipeline

### Premise

Students continue with the **same APP example they chose in Session 1**.
Session 2 is "upgrade your session-1 result" — flow head, learned
summary if not already, SBC diagnostics, optional sequential round. Each
addition is motivated by a specific failure of the previous version, so
the lecture content earns its place in front of them.

Strong ownership (each student walks out with one polished notebook on
their chosen problem); no new simulator setup overhead; clean comparison
("my baseline" vs "my flow pipeline") that is theirs personally.

### Five-block structure (~120 min)

#### Block 1 — Recap and diagnose the baseline (~15 min)

- Reload session-1 result on the chosen APP example.
- Plot posterior against truth. Identify where the Gaussian-head version
  is wrong: multimodality? skew? miscalibration?
- This is the lecture failing on their own code, on purpose.

#### Block 2 — Swap to a flow head (~30 min)

- Use `sbi` (or `lampe`) to replace the Gaussian head with a normalising
  flow (neural spline flow or affine coupling).
- Retrain on the same simulator output.
- Compare posteriors side-by-side with the Gaussian-head baseline.
- Observe: posterior now captures the actual shape.

#### Block 3 — Neural summary network (~30 min)

If session 1's example used a hand-built summary (GW with band powers,
CR with histogram), now replace with a learned summary network. Image
students (Option C) already have one; for them this block is "improve
the architecture" or skip to Block 4.

- Plug a small CNN / MLP between observation and inference head.
- Retrain end-to-end. Compare to Block 2's posterior.

#### Block 4 — SBC and coverage diagnostics (~30 min)

- Run simulation-based calibration (SBC).
- Plot rank histograms; check expected coverage.
- If miscalibrated, discuss why (overconfidence is the usual culprit).
- This is where the diagnostics lecture pays off; without this block,
  the room would not know whether they should *trust* their session-1
  posterior, which is the question that matters most in practice.

#### Block 5 — Optional stretch (~15 min)

Pick one:

- **Sequential NPE:** one round of training on simulations near x_obs;
  observe posterior tighten.
- **Flow matching head:** swap the flow for an FM head from `sbi` or
  `zuko`, compare. (Aspirational; depends on library support at school
  time.)
- **Compare to MCMC reference:** for students who want a sanity check,
  run an MCMC on a tractable likelihood version (if available) and
  overlay.

### Wrap (joint, ~5 min)

Joint plot per student: ABC concept (from Lec 1b) vs Gaussian-head NPE
(Session 1) vs flow-based NPE + summary net + diagnostics (Session 2).
The room sees their full school in one figure.

### Decisions locked in

- **Continuity by simulator.** Each student stays with their Session 1
  APP example through Session 2. No new sims.
- **Library: `sbi`.** Best documentation, biggest community, most likely
  to still exist and be maintained at school time. Alternatives
  (`lampe`, `zuko`, `BayesFlow`) noted but not the default.
- **Diagnostics block is non-negotiable.** SBC is the question students
  will be asked when they go home and try this on real data. Make it
  ordinary, not exotic.
- **Sequential and FM are stretch goals.** Mentioned, optional.

---

## Cross-session design notes

### Hardware

Everything must run on a laptop CPU in <5 min per training run.
Constrain network sizes and dataset sizes accordingly. Provide a hosted
Colab fallback for students whose local environments break.

### Authoring workflow

The three APP-example notebooks are the biggest authoring task. Workflow
once we start building:

1. **CW writes a one-paragraph spec per example** (simulator details,
   parameters, observable, suggested architecture). Five minutes of
   focused time per example.
2. **Claude drafts the full notebook** (simulator code, scaffolding,
   exercise cells, reference posterior via fine-grid or emcee,
   solutions). ~30-60 min agent time per example, can run in background.
3. **CW reviews one notebook at a time**, flags physics/pedagogy
   issues; Claude revises.
4. Total CW time per example: ~1-2 h end-to-end. Three examples ≈ one
   focused day, not three.

Suggested starting points to reduce authoring effort:

- **GW:** strip down `dingo` or `sbi`-benchmark GW example.
- **CR:** lean on published `sbi` benchmarks or swyft tutorials.
- **Point sources:** borrow from swyft/peregrine point-source population
  inference work; BayesFlow has image examples too.

### Calibration notes for the author (Claude)

- Claude is stronger on GW and point-source-on-image problems than on
  CR spectra. GW: canonical waveform models exist in libraries. Point
  sources: textbook Poisson + PSF. CR: wider design space, more spec
  from CW needed to converge on a defensible problem.
- All three problems should have a *tractable reference posterior* so
  students can check their answer. Low-D problems → fine grid; modest-D
  → emcee.
- Each notebook should end with the same final plot template (posterior
  + truth overlay + summary metrics) so cross-comparison is visually
  consistent at the wrap.

### What is explicitly *not* in either session

- Classical ABC implementation (Lec 1b handles).
- NRE / NLE hands-on (Lec 3 mention only).
- Flow matching / diffusion hands-on (Lec 3 visualised only; FM appears
  as Session 2 stretch goal at most).
- Embedding-network architecture deep dive (Lec 4 lecture content;
  hands-on uses pre-built nets).
- Anything that requires GPU.
