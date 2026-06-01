# Lecture Plan

## Lec 1a: Overview of ML in APP and GW

### Content

- This should be a list of recent applications of ML and AI technology to APP and GW research problems. Ideally this will constitute an exhaustive list of different types of use cases, with concrete recent examples.
- Examples should include the loudest examples (works with deep mind about GWs in Virgo(?), work with mistral if any, analysis of actual real world data like data from IceCube, data reduction, but also parameter inference in gravitational waves, e.g. DINGO and others).
- The entire lecture can start with a single history slide going back as far as possible regarding statistical data analysis in astronomy/astrophysics/etc.
- The last block of this should be about simulation-based inference, essentially all examples where model-based parameter inference happens. I need to clearly separate here likelihood-based inference (MCMC on steroids) and SBI (ABC on steroids), as segway to then focus on SBI.

### Construction plan

- Constraints: About 15 slides (=45 minutes); style close to existing lecture html files; 1-2 figures per slide extracted from existing papers (which you download before, but in many cases you can just access the html version via arxiv) + bullet points with key messages). At least one reference per slide.
- Approach: We start with outlining structure and general topics, then collect relevant pdfs in a folder (or just links to webpages of the articles), then you extract all information from there and make a first draft in a more detailed md file, which I iterate over, then you create the final reveal.js presentation based on the plan.
- Nice to have: Maybe we can also run this through zotero? With all papers on zotero and you just extract information directly from there? Would ease reading. You could also highlight the figures that were extracted directly in the zotero API.

## Lec1b: Classic ABC (45 min)

### Narrative arc

A three-demo staircase, each demo solving one problem of the previous one, ending on the problems neural SBI exists to solve.

1. **Rejection ABC** — establish the method and its scaling pain.
2. **+ Summary statistic** — show how dramatically a good summary helps.
3. **KDE in (θ, x)-space** — show what *actually* solves the ε problem in 1D, and why it doesn't scale.
4. **Challenges → NNs** — name the two failures (summary choice, density estimation in high D) and map each to a later lecture in the school.

### Running example: one-parameter ball throw

- Parameter: launch angle θ (1D, uniform prior over a sensible range).
- Observation: landing position x (1D).
- Simulator: deterministic physics + Gaussian noise.
- Reason for 1D/1D: posteriors can be plotted live and overlaid with the analytic / fine-grid truth. Resist any temptation to add a second parameter in this lecture; 2D ABC demos look muddy and steal time from the punchline.

### Slide-level outline

1. **What is ABC.** Pseudocode in three lines: sample θ \~ p(θ); simulate x \~ p(x|θ); accept if d(x, x_obs) &lt; ε. Picks up directly from 1a's closing slide.
2. **Demo 1 — Rejection ABC.** Sliders: **N** (total sims) and **M** (keep top-M closest to x_obs). Use M/N instead of ε so the plot is never empty. Live posterior is a histogram of accepted θ's. Overlay analytic truth.
   - Key message: small M → noisy posterior; large M → prior leakage; you need N huge to have enough sims near x_obs. Wasted simulator calls.
3. **Demo 2 — Add a summary statistic.** Throw n_balls per θ, summary = mean landing position. Same N, M sliders, plus n_balls. Compare to demo 1 side-by-side: posterior tightens dramatically at the same N.
   - Key message: a well-chosen summary buys orders of magnitude in sample efficiency. (Open question for later: who chooses the summary?)
4. **Demo 3 — KDE in (θ, x)-space.** No accept/reject. Build a 2D KDE on all N pairs (θ\_i, x_i), draw a vertical line at x_obs, project the conditional slice as the posterior. Sliders: **N** and bandwidth **h**.
   - Key message: uses *all* sims, no ε to tune, recovers a clean posterior at much smaller N. This *is* the ε → 0 limit done right.
   - Failure mode shown on the *same* demo or a frozen follow-up slide: add a dummy extra x-component (or just state "now imagine x is 100-D") so the bandwidth collapses and the estimate degrades. This is the curse of dimensionality, and it sets up the challenges slide.
5. **Challenges slide — explicit syllabus.** Two named failures, each pointing at a later lecture:
   - **Summary identification** (when x is high-dimensional, structured, or unobvious) → neural summary networks / classifiers.
   - **Density estimation in high D** (curse of dimensionality on KDE) → normalising flows, NPE / NLE.
   - End-to-end: one network learns both → the rest of the school.

### Decisions locked in

- **No θ-space-only smoothing toggle.** Dropped. KDE/Gaussian on the accepted cloud doesn't solve anything ABC struggles with and risks confusing students about what the real bridge to neural SBI is.
- **M/N, not ε**, as the live slider. ε stays in the pseudocode slide.
- **Three demos, no more.** Resist a fourth.
- Each demo: \~6–8 min, no live coding.

### Construction plan

- Plan is now frozen here. Next: update the TASI lecture notes (separate PDF) to include these three examples, plus ipynb sources for the illustrating plots.
- From the TASI notes, their references, and the ipynbs, build the reveal.js deck — including JS reimplementations of the three ABC demos (animations + sliders, matching the style of the existing lecture\*html files).

## Lec 2: Building q\_φ(θ|x) — from chi² to neural densities (90 min)

### Narrative arc

One continuous story: parameterise p(θ|x), make it more flexible as need demands. Every ingredient (basis functions → SGD → NNs → heteroscedastic noise → flows) enters because the *previous* model is too restrictive.

1. **Reframe chi² as conditional Gaussian density estimation.** Reveal the role-swap: from "x → point estimate of θ" to "x → density q(θ|x)."
2. **Show the basis-function problem live.** Overfitting animation as the motivation to learn the basis instead of choosing it.
3. **NNs + SGD as the answer.** Learn μ(x) as a neural net, optimised by GD because the closed form is gone.
4. **Generalise the noise model.** σ²(x) (heteroscedastic), then Gaussian too restrictive (multimodal) → preview normalising flows.
5. **Handoff to Lec 3.** We now have q\_φ(θ|x). Next: train it on simulator output → NPE.

### Audience calibration

Students have basic stats, Bayes' theorem at the formula level, chi²-fitting as a procedure, maybe MCMC. They have likely *not* seen:

- chi² derived explicitly from a Gaussian log-likelihood (they use it, they don't write it as −2 log L)
- "MLE" as a named concept distinct from "fitting"
- The role-swap from estimating θ to modelling q(θ|x)
- Linear-basis notation μ(x; w) = Σ w_i φ\_i(x) as a modelling choice
- Anything substantive about neural networks, SGD, or autograd
- Conditional density estimation as a framework
- Normalising flows

So Block A is *fast review of the math* but *careful introduction of the framing*. Blocks C and D are genuinely new material and get the most time.

### Slide-level outline (\~26 slides, \~3.5 min each)

**Block A — chi² as Gaussian conditional density estimation (\~5 slides**)Source: largely Lec 1 §7–§15, reframed.

1. **What chi²-fitting really is.** chi² = −2 log L under Gaussian noise. "Minimise chi²" = MLE. Introduce the right-hand side; left-hand side is review.
2. **The linear-basis picture.** μ(x; w) = Σ w_i φ\_i(x). Polynomial, Gaussian RBF. The choice of {φ\_i} is the modelling decision; w is just MLE. (Lec 1 §9–§10.)
3. **The role-swap — the most important slide of the lecture.** Until now: x → point estimate of θ. New view: x → q(θ|x) = N(μ(x), σ²). The output is a *density*, not a point. Clean visual: scatter of (θ, x) pairs + fitted Gaussian band around μ(x). (Reframe of Lec 1 §13.)
4. **Closed-form MLE for w; residual variance gives σ².** Mention not derive. Flag explicitly: σ² is *constant* here — same uncertainty everywhere. Revisit later. (Lec 1 §15, condensed.)
5. **Worked example.** Polynomial μ(x) on a 1D dataset, Gaussian band shown. (Lec 1 §19, the interactive.)

**Block B — Overfitting and the basis-function problem (\~4 slides**)Source: Lec 1 §19–§24. Keep the animation; it does work no static slide can.

6. **Overfitting animation.** Lec 1 §19, live: too-low / right / too-high polynomial degree. Watch the curve wiggle to chase noise.
7. **Underfit / good fit / overfit visual triplet.** (Lec 1 §21–§23, compressed to one slide if possible.)
8. **Validation.** Train/val split; the U-curve. (Lec 1 §24.)
9. **The basis-function problem.** Underfit → too few; overfit → too many, wrong shape. *Can we learn the basis functions themselves?* (Bridge slide, mostly reused from Lec 3 §23 — "The φ Problem.")

**Block C — Learn the basis functions: NNs + SGD (\~10 slides**)Source: Lec 3 §24–§35, plus one slide from Lec 4. Skip Lec 3's classification first half entirely.

10. **Towards learnable building blocks.** (Lec 3 §24–§25.)
11. **The artificial neuron + common activations.** (Lec 3 §27 + §28 combined.)
12. **One hidden layer = learnable φ.** (Lec 3 §29.) This is the payoff of §9's question.
13. **Universal approximation — one line.** (Lec 3 §32, no proof.)
14. **No closed form anymore → enter gradient descent.** Motivate GD by the fact that we want learned features. (Lec 3 §13 reframed.)
15. **GD details + pseudocode.** (Lec 3 §14.)
16. **Mini-batch SGD.** (Lec 3 §17.)
17. **Backprop — one slide, intuitive.** Computation graph picture; "autograd does this for you." No chain-rule derivation. (Lec 4 §3 + caption.)
18. **One bullet on Adam, one on learning-rate schedule.** Mention, don't teach. (Lec 4 §28 condensed.)
19. **MLP-regression in action.** Live demo, learnable μ(x) fitting the same 1D dataset as slide 5. (Lec 3 §34.)

**Block D — From Gaussian to flexible q\_φ(θ|x) (\~6 slides**)Source: Lec 6 §13–§14 and §19–§20, plus one new slide.

20. **Bayes-optimal classifier ↔ sigmoid — one slide.** Connects classification to posterior estimation. Lets Lec 3 mention NRE in one sentence later. (Compressed from Lec 3 §7.)
21. **Heteroscedastic noise — σ²(x).** Both μ(x) and σ²(x) as NN outputs. Same architecture, two heads. (New slide.)
22. **Gaussian-head q\_φ(θ|x) — the destination.** Reframe of Lec 6 §13: "Example head 2: Gaussian-head NPE" — but in Lec 2 we present it as supervised learning of p(θ|x) on (θ, x) pairs, not yet as NPE.
23. **Live demo on a sine curve.** Lec 6 §14, reframed: Gaussian-head NN on (θ, x) pairs from a 1D simulator.
24. **Why we need flows — Gaussian is too restrictive.** Multimodal posteriors break the Gaussian assumption. (Lec 6 §20, teaser only.)
25. **Normalising flows — one-slide preview.** (Lec 6 §19, names + picture, no change-of-variables yet — that's Lec 3.)

**Block E — Handoff (\~1 slide)**

26. **Where we are; where we're going.** We have q\_φ(θ|x). Lec 3: train it on simulator output (θ\_i, x_i) \~ p(θ)p(x|θ) → neural posterior estimation. Flows get their full treatment there.

### Decisions locked in

- **Notation: use (θ, x) from slide 1.** Existing Lec 1 uses (x, y) for regression. Global find-and-replace during the retrofit; each slide needs a notation check.
- **Drop classification as a standalone block.** One slide acknowledgement (slide 20). NRE is not the destination; NPE is.
- **Backprop: one intuitive slide, no chain-rule derivation.** Audience is PhD physicists; the Lec 4 derivation is calibrated for an undergrad course and would be condescending.
- **No Lec 4 training-dynamics content.** Initialization, layer norm, double descent, spectral bias — all dropped. Adam gets one bullet.
- **No Lec 4b.** Dropped entirely; weight decay can be a side-bullet if at all.
- **Flows: preview only in Lec 2.** Full change-of-variables, coupling layers, log-Jacobians — Lec 3.
- **Overfitting animation stays.** Pedagogical work no static slide can do.

### Construction plan

- Retrofit existing slides rather than draft from scratch. \~80% of content is already drafted across Lec 1, Lec 3, Lec 4, Lec 6.
- Pass 1: re-title and re-caption to the new framing (chi² → q(θ|x), point estimate → density).
- Pass 2: notation sweep — (x, y) → (θ, x) — slide by slide.
- Pass 3: write the genuinely new slides (3, 9, 21, 26).
- Pass 4: integration check — does Block A's role-swap slide actually prepare slide 22? Does Block C's MLP demo (19) visually match Block D's Gaussian-head demo (23)?

## Lec 3: Inference heads — flows, flow matching, diffusion (90 min)

### Narrative arc

Lec 3 is the **inference-head** lecture of the school's organising triptych:

simulations → compression → inference head

The triptych is introduced in slide 2 and re-appears in Lec 4 (compression) and Lec 5 (diagnostics). Lec 3's job is to populate the inference-head slot: given a (good) summary of x, what are the choices for putting a density on θ?

1. **Ratify NPE.** What we built in Lec 2 (supervised q\_φ(θ|x)) becomes NPE the moment the data come from a simulator. The conceptual leap is small; say it explicitly and move on.
2. **Name the variants briskly.** NRE (classifier angle), NLE (model the forward density). Comparison slide. NPE remains the spine.
3. **Spend the lecture on the flow head.** Full machinery: change of variables, Jacobians, coupling layers, conditional flows, 2D demo.
4. **Add flow matching and diffusion as visualised side-companions.** No derivations. Paired via the unified continuous-time picture (FM = ODE, diffusion = SDE).
5. **Application showcase + handoff to Lec 4.** LISA, DINGO. Embedding teaser signposts "Lec 4 = compression."

### Anti-overlap discipline

Already done in Lec 1+2 — do *not* re-do:

- The SBI framing (simulator vs likelihood; the inference problem formally) — Lec 1a's last block.
- The simulator-as-data idea — Lec 1b's spine.
- "Train q\_φ on (θ, x) pairs by NLL" — Lec 2 Block D. The loss is literally the same; only the data source (simulator vs fixed set) changes. One slide ratifies this; do not re-derive.
- Gaussian-head NPE as a worked example — Lec 2 slides 22-23.
- Why we need flows (multimodal posteriors) — Lec 2 slides 24-25.

These show up in Lec 3 only as one-line callbacks, never as fresh content.

### Slide-level outline (\~24 slides, \~3.75 min each)

**Block A — NPE and the inference-head systematics (\~4 slides)**

1. **Recap of Lec 2 + the leap.** "We trained q\_φ(θ|x) on (θ, x) pairs by NLL. If those pairs come from a simulator (θ\_i \~ p(θ), x_i \~ p(x|θ\_i)), we just built NPE." One slide; no re-derivation.
2. **The SBI triptych: sims → compression → inference head.** *New slide.* Introduces the organising principle for Lec 3 (inference head) and Lec 4 (compression). Visual: pipeline diagram with three boxes.
3. **NRE — the classifier angle.** Cashes in Lec 2 slide 20 (Bayes-optimal classifier ↔ sigmoid). Train a classifier on (θ, x) vs (θ, x') pairs; the learned logit *is* the log-ratio log p(θ|x)/p(θ). One slide.
4. **NLE — the dual.** Model p(x|θ) instead; recover the posterior via MCMC at the end. One slide.
5. **NPE / NRE / NLE comparison.** When each shines; which one DINGO uses; amortisation trade-offs. One slide. (Block A is then 5 slides; accept that.)

**Block B — Inference heads in depth (\~14-15 slides)**

*B1 — Flows (the deep treatment, \~10 slides)*. Source: Lec 6 §21-29.

 6. **Why we need flows beyond Gaussian-head.** One-line recap of Lec 2 slide 24; jump in. (Lec 6 §20, compressed.)
 7. **Change of variables.** (Lec 6 §21.)
 8. **Worked example: warping a Gaussian.** (Lec 6 §22.)
 9. **Normalising flow = stack of invertible maps.** (Lec 6 §23.)
10. **Log-density by chain of Jacobians.** (Lec 6 §24.)
11. **Affine coupling layers.** (Lec 6 §25, possibly compressed.)
12. **Conditional flows — plugging in the data.** (Lec 6 §26.)
13. **Flow pushforward — 2D worked example.** (Lec 6 §27.)
14. **Flow pushforward — live 2D demo.** (Lec 6 §28.)
15. **NPE-with-flows: the full pipeline.** (Lec 6 §29, reframed to emphasise this is the *inference head* slot of the triptych.)

*B2 — Flow matching and diffusion (the visualised side-companions, \~4 slides)*. All new slides; visualisations and one-line ideas, no derivations.

16. **Continuous-time generative models — unified picture.** One panel with smooth ODE arrows (FM), one with noisy SDE arrows (diffusion). Same trajectory frame; one deterministic, one stochastic.
17. **Flow matching.** Learn v\_φ(θ, t) such that the ODE transports prior → posterior. One-line training objective; one-line sampling procedure. References: Lipman et al. 2023, recent SBI applications.
18. **Diffusion / score-based generative models.** Learn the score ∇log p_t(θ|x); sample by reverse SDE. References: Sohl-Dickstein, Song & Ermon, Ho et al.; score-based SBI applications (Glaser et al., posterior-diffusion GW papers).
19. **When to reach for which.** Flows = exact density, slower sampling; FM/diffusion = fast sampling, no exact density (well, with effort). One pragmatic table.

**Block C — Application showcase + handoff (\~4 slides)**

20. **LISA stochastic GW background.** (Lec 6 §30.)
21. **DINGO for binary-merger PE.** *New slide.* Direct evidence that NPE with flows works in production GW analysis. The hook for the room.
22. **Embedding network teaser.** (Lec 6 §16, retitled.) "We've been assuming a good summary of x — how do we *get* one? That's Lec 4."
23. **Summary + Lec 4 preview.** (Lec 6 §31, reframed.)

### Decisions locked in

- **Inference-head framing is the organising principle.** Triptych introduced in slide 2 and re-used in Lecs 4 and 5.
- **NPE stays the spine; NRE and NLE are background colour.** One slide each, then back to NPE.
- **Flows get the deep treatment, FM and diffusion are visualised side-companions.** No derivations of FM or diffusion; the continuous-time picture is the visual anchor.
- **FM and diffusion paired in one unified visual** (ODE vs SDE trajectories), not introduced as two unrelated topics.
- **DINGO gets its own slide.** The school is APP/GW-flavoured; the loudest production example deserves explicit treatment.
- **No re-derivation of the NLL loss, the simulator setting, or Gaussian-head NPE.** These are Lec 1+2 work and appear only as one-line callbacks.

### Construction plan

- \~80% of Block B1 (flows) is direct lift from Lec 6 §21-29.
- New slides to draft: slides 2 (triptych), 3-5 (NRE/NLE/comparison), 16-19 (FM + diffusion + when-to-use), 21 (DINGO).
- Application slide 21 needs a recent DINGO figure (arXiv or published plot) — pull during the figure-collection pass.
- FM and diffusion slides need one canonical figure each; Lipman et al. and Song et al. both have good open-license figures.
- Sanity check: the continuous-time unified visual (slide 16) must be drawn carefully — it's doing structural work in the lecture.