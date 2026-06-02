# Lec 1a — Annotated reference longlist

Target: 45 min, ~15 slides, 1–2 figures per slide, ≥1 reference per slide. Time window: 2018–2026 emphasis, with pre-2018 anchors only for the history block.

**Verification key:**
- ✓ = primary source verified 3-0 by the deep-research workflow (claim, authors, headline numbers checked against arxiv abstract).
- ◇ = surfaced by the research scan, primary URL in hand, but **not** adversarially verified. Skim before quoting numbers.
- ✗ = no source found by the scan; entry below is from background knowledge or marked TODO. Verify before using.

The plan asked us to be explicit about loud-but-unconfirmed examples. The DeepMind/LIGO paper is real and verified ✓. A Mistral-in-astro paper of the kind the brief described **was not found** by the scan and should be presented as "no such paper to my knowledge as of 2026-06" rather than fabricated.

---

## Slide structure (locked 2026-06-02)

Spine: the **scientific method as a loop**, with ML at every arrow:

```
   theory ─▶ model/simulator       (Block 3)
   instrument ─▶ data              (Blocks 4–5)
   data + model ─▶ inference ─▶ insight   (Block 6)
```

Block 5 (data) follows Block 4 (instrument), reading naturally left-to-right along the second arrow. Block 6 is the heaviest; it sets up the rest of the school. Slide 1 shows the three rows visually, the rest of the lecture walks them.

### 15-slide outline

**Opening (1 slide)**
1. **The scientific method loop.** Three rows: theory→model, instrument→data, data+model→inference→insight. APP+GW keyword clusters under each box. Verbal framing: "ML lives at every arrow; this lecture walks all of them, but the bottom row is where we'll spend most time and where the rest of the school picks up."

**Block 1 — History (1 slide)**
2. Gauss/Ceres (1809) → Hertzsprung-Russell (1911–13) → Schmidt V/Vmax (1968) → SExtractor (Bertin & Arnouts 1996) → Sloan-era ML (Ball & Brunner 2010) → deep-learning era (Dieleman 2015 Galaxy Zoo CNN). One composite figure. *(2nd-pass scan in progress.)*

**Block 2 — ML for theory (1 slide, NEW)**
3. Symbolic regression and equation discovery. **Star pick:** Cranmer et al. 2020 "Discovering symbolic models from deep learning with inductive biases" (graph nets → force laws / DM halo overdensity formula). Supporting: PySR, AI Feynman; one-line on AI-scientist agents. Honest framing: thin in APP/GW specifically, but the arrow exists. *(2nd-pass scan in progress.)*

**Block 3 — ML for the simulator (2 slides)**
4. Cosmology emulators. **Star pick:** CosmoPower (Spurio Mancini 2021) + one line on CAMELS (Villaescusa-Navarro 2021) as the training-set backbone. ◇
5. GW waveform surrogates. NRSur7dq4 (Varma 2019) as the anchor; mlgw and 2023–26 neural-ODE / diffusion follow-ups. *(2nd-pass scan in progress.)*

**Block 4 — ML for the instrument (1 slide)**
6. **Deep Loop Shaping** (DeepMind + LIGO Instrument Team, Science 2025). RL controlling mirror suspensions at LIGO Livingston, >30× noise reduction in 10–30 Hz. ✓ The loud slide.

**Block 5 — ML for the data (4 slides)**
7. GW glitches: Gravity Spy (Zevin 2017 + Wu 2024 O4 update). ✓
8. Neutrino reconstruction: IceCube hexagonal-kernel cascade CNN (Abbasi 2021); DeepCore symmetry-exploiting CNN (2023). ✓
9. Galaxy morphology: Galaxy Zoo Bayesian CNN (Walmsley 2020). ✓
10. γ-ray source classification: Fermi-LAT voting ensemble (Zhu 2023). ✓

**Block 6 — ML for the inference (4 slides)**
11. GW detection nets: Gabbard 2018 (historical) → AresGW 2024 (current, real O3 noise). ✓ One slide, before/after.
12. GW parameter inference: DINGO + DINGO-IS (Dax 2021, 2022). ✓ The headline result.
13. Inference beyond GW: strong lensing / 21cm / CMB. One example. *(2nd-pass scan in progress.)*
14. **The two inference families + SBI taxonomy.** Likelihood-on-steroids (CosmoPower-HMC pipelines) vs SBI (ABC-on-steroids). NPE/NLE/NRE figure (Cranmer-Brehmer-Louppe 2020). One dense slide. ◇

**Closing (1 slide)**
15. **Two transverse trends + segue to Lec 1b.** Foundation models touching astro (AstroCLIP, AstroLLaMA; explicit "no Mistral-in-astro yet"). Then: pseudocode for ABC as the bridge to Lec 1b. ◇

### Slide-to-reference index

| #  | Slide                                         | Star pick                                                   | Status |
|----|-----------------------------------------------|-------------------------------------------------------------|--------|
| 1  | Scientific method loop (opener)               | —                                                           | n/a    |
| 2  | History arc                                   | SExtractor (Bertin & Arnouts 1996) + Gauss anchor           | ✓ post-1990, ✗ pre-1990 |
| 3  | ML for theory (symbolic regression)           | Cranmer et al. 2020 graph-nets → symbolic                   | ✓      |
| 4  | Cosmology emulators                           | CosmoPower (Spurio Mancini 2021) + CAMELS                   | ◇      |
| 5  | GW waveform surrogates                        | NRSur7dq4 (Varma 2019) + DANSur (Fernandes 2024)            | ✓      |
| 6  | LIGO interferometer control                   | Deep Loop Shaping (Buchli et al., Science 2025)             | ✓      |
| 7  | GW glitches                                   | Gravity Spy (Zevin 2017 + Wu 2024)                          | ✓      |
| 8  | Neutrino reconstruction                       | IceCube cascade-CNN (Abbasi 2021)                           | ✓      |
| 9  | Galaxy morphology                             | Walmsley 2020 Bayesian CNN                                  | ✓      |
| 10 | γ-ray source classification                   | Fermi-LAT ensemble (Zhu 2023)                               | ✓      |
| 11 | GW detection nets                             | Gabbard 2018 + AresGW 2024                                  | ✓      |
| 12 | GW parameter inference                        | DINGO + DINGO-IS (Dax 2021, 2022)                           | ✓      |
| 13 | Inference beyond GW                           | Wagner-Carena et al. 2023 (real HST lenses) — recommended   | ◇ pending verify |
| 14 | Two inference families + SBI taxonomy         | Cranmer-Brehmer-Louppe 2020                                 | ◇      |
| 15 | Foundation models + segue to Lec 1b           | AstroCLIP (Parker 2024) + AstroLLaMA (Nguyen 2023)          | ◇      |

After second-pass scan (2026-06-02): **11 of 14 content slides** have verified anchors. Three slides have scan-only references (4, 14, 15). One slide (13) has a recommended star pick but still needs adversarial verification. Pre-1990 history anchors (Gauss, Hertzsprung-Russell, Schmidt) are textbook-standard but not re-verified; a quick ADS pass before slide finalisation is fine.

---

## Second-pass results (2026-06-02)

### Block 1 — History anchors (verified)

1. ✓ **Bertin & Arnouts 1996, "SExtractor: Source extractor"** — A&AS 117, 393 (DOI 10.1051/aas:1996164). The widely-deployed 1990s astronomy pipeline with `CLASS_STAR`, a supervised multilayer feed-forward NN trained on ~600 simulated images for star/galaxy separation. The "ML existed in astro long before deep learning" anchor. **Figure:** `CLASS_STAR` stellarity histograms, Sect. 4.

2. ✓ **Ball & Brunner 2010, "Data mining and machine learning in astronomy"** — IJMPD 19, 1049 (arxiv:0906.2173). The canonical Sloan-era ML review: ANN, SVM, photo-z, classification, time-domain. **Figure:** photo-z scatter plots and ANN/SVM schematics.

3. ✓ **Dieleman, Willett & Dambre 2015, "Rotation-invariant convolutional neural networks for galaxy morphology prediction"** — MNRAS 450, 1441 (arxiv:1503.07077). The deep-learning era marker for astronomy: won the Galaxy Zoo Kaggle challenge, >99% reproduction of crowd consensus on high-agreement images. **Figure:** rotation-invariant architecture diagram + per-question accuracy.

4. ✗ **Pre-1990 anchors** still to confirm against ADS: Gauss 1809 *Theoria Motus*; Hertzsprung 1911 / Russell 1913 (use Russell 1914 *Popular Astronomy* for the citation, the original HR diagram paper); Schmidt 1968 *ApJ* 151, 393 (V/Vmax). Uncontroversial textbook citations, but worth a 5-minute ADS pass before slide ships.

### Block 2 — ML for theory (verified)

1. ✓ **Cranmer, Sanchez-Gonzalez et al. 2020, "Discovering symbolic models from deep learning with inductive biases"** — NeurIPS, arxiv:2006.11287. **Star pick.** GNNs + symbolic regression rediscover Newtonian force laws from N-body sims, then derive a *new* analytic formula for dark-matter overdensity δ (MAE 0.088). **Wording care:** the discovered formula predicts overdensity δ, *not* NFW concentration c — track the paper's wording on the slide. **Figure:** dark-matter overdensity scatter vs GNN/SR prediction; N-body force-law message-function plots.

2. ✓ **Cranmer 2023, "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl"** — arxiv:2305.01582. The open-source SR library reference; Julia backend, multi-population evolve-simplify-optimize. **Figure:** Pareto-front complexity-vs-loss plots.

3. ✓ **Udrescu & Tegmark 2020, "AI Feynman"** — Science Advances (arxiv:1905.11481). Recursive physics-inspired SR (NNs + symmetry + separability + dimensional analysis) recovers 100 Feynman-textbook equations. Historical anchor. **Figure:** recovery-rate bar chart over Feynman dataset; recursive decomposition diagram.

4. ✓ **Bartlett, Desmond & Ferreira 2022, "Exhaustive Symbolic Regression"** — arxiv:2211.11461. ESR applied to cosmic chronometers + Pantheon+ to reconstruct H(z); ~40 of 5.2M candidate functions fit better than the Friedmann equation under MDL. Clean cosmology-adjacent SR application. **Figure:** H(z) Pareto front; symbolic-expression ranking vs Friedmann.

5. ✓ **Lemos et al. 2022, "Rediscovering orbital mechanics with machine learning"** — MLST 2023, arxiv:2202.02306. GNN trained on 30 years of *real* solar-system trajectories, SR recovers Newton's law of gravitation plus solar-system masses without prior specification. The "ML rediscovers physics from observation" demonstration. **Figure:** inferred-vs-true mass plot; recovered force-law functional form.

6. ✗ **Lu et al. 2024 Sakana "AI Scientist" (arxiv:2408.06292)** — flagged but **not verified** in this pass. Cite only after a 5-minute arxiv check; no astro-specific "AstroPilot" autonomous-agent paper was confirmed to exist.

### Slide 5 — GW waveform surrogates (verified)

1. ✓ **Varma et al. 2019, "Surrogate models for precessing binary black hole simulations with unequal masses" (NRSur7dq4)** — PRR (arxiv:1905.09300). **Star pick / NR-surrogate anchor.** 1,528 NR simulations, q ≤ 4, χ ≤ 0.8, generic spins. ≥1 order of magnitude more accurate than alternatives *within training range* (preserve the scope qualifier on the slide). Used in GW190521. **Figure:** mismatch-vs-alternatives plots; precessing waveform overlays.

2. ✓ **Schmidt et al. 2020, "mlgw: a machine learning surrogate model for gravitational waves"** — arxiv:2011.01958. PCA + regression on ~1000 TEOBResumS / SEOBNRv4 EOB waveforms; ~10⁻³ faithfulness, 10–50× speedup over EOB. Aligned-spin BBH, q = 1–20. **Figure:** faithfulness histograms; EOB-overlay.

3. ✓ **Tissino et al. 2022, "mlgw-bns"** — arxiv:2210.15684. BNS counterpart trained on TEOBResumSPA, frequency-domain, ~35× speedup, validated on GW170817 parameter estimation. The natural BNS extension. **Figure:** GW170817 posterior overlays vs TEOBResumSPA.

4. ✓ **Fernandes et al. 2024, "DANSur"** — arxiv:2412.06946 (PRD 112, 043026, 2025). **Star pick for "modern NN waveform surrogate."** Dual-stage NN: pretrained on approximant waveforms, fine-tuned with NR. Generates millions of waveforms in <20 ms on GPU, mean NR mismatch ~10⁻⁴. **Figure:** throughput-vs-accuracy comparison; NR mismatch distribution.

5. ✓ **Liao & Lin 2021, "Generative model for gravitational waveforms with conditional autoencoder"** — PRD 103, 124051 (arxiv:2101.06685). Conditional autoencoder for BBH inspiral-merger, q = 1–10; >97% matched-filter overlap, 10–100× speedup over EOBNR. Useful if the slide wants a generative-model framing. **Figure:** schematic + overlap histograms.

**Slide framing note:** if the lecture emphasises "ML strictly," lead with DANSur. If it emphasises "surrogate modelling broadly," lead with NRSur7dq4 and mention DANSur as the modern NN-native heir. The latter reads more naturally given the audience.

### Slide 13 — End-to-end inference beyond GW (recommended shortlist, NOT yet verified)

This gap was not covered by the second-pass verified claim set. The workflow returned a **recommended shortlist** that matches the slide's needs; each needs a primary-source check before the slide ships.

1. ◇ **Wagner-Carena et al. 2023** — arxiv:2203.00690. **Recommended star pick.** Hierarchical SBI on real HST strong lenses. The cleanest "real instrument data, here is the posterior" candidate for this slide. *Verify before quoting.*

2. ◇ **Hezaveh, Levasseur, Marshall 2017** — Nature 548, 555 (arxiv:1708.08842). Historical anchor: fast automated strong-lens analysis with CNNs on real HST cutouts. *Verify.*

3. ◇ **Coogan, Karchev, Weniger 2022** — arxiv:2010.07032. Dark-substructure SBI in strong lenses. Natural self-cite from the lecturer's own group. *Verify.*

4. ◇ **Saxena et al. 2023** — arxiv:2303.07339. 21cm SBI applied near-HERA data. *Verify.*

Decision-needed: pick *one* anchor for this slide. Strong-lensing-on-HST is the most visually satisfying ("here is the lens image, here is the posterior on substructure mass") and has three of the four candidates. Wagner-Carena 2023 looks like the right call modulo verification.

---

## Open items before slide drafting

1. **Slide 13 verification pass** — confirm Wagner-Carena 2023 numbers and figure clarity before adopting as star pick. ~5-10 minute job.
2. **Pre-1990 history anchors** — quick ADS pass to lock exact citation format for Gauss 1809, Russell 1913/14, Schmidt 1968.
3. **Slide 4 emulator slide** — CosmoPower headline numbers (◇) still need verification before quotable on slide.
4. **Slide 14 SBI taxonomy** — same for Cranmer-Brehmer-Louppe 2020 (◇).
5. **Slide 15 foundation models** — AstroCLIP / AstroLLaMA (◇) numbers + figure choice.
6. **"AI Scientist" Sakana (Lu et al. 2024)** — confirm arxiv:2408.06292 before mentioning anywhere on slide 3.

---

## Block 1 — History anchors (one slide) ✗ TODO

The research scan returned nothing for this block; the items below are from background knowledge and **need a verification pass** before going on a slide.

1. **Gauss 1809, *Theoria Motus*** — least squares applied to rediscovering Ceres after Piazzi's gap. The "ML-before-ML" origin point worth a one-line callout.
2. **Hertzsprung 1911, Russell 1913** — independent statistical population studies of stellar luminosities and colours; arguably the first scatter-plot-as-discovery in physics.
3. **Schmidt 1968, "Space distribution and luminosity functions of quasi-stellar radio sources"** — V/Vmax test; statistical inference on a real survey.
4. **Bertin & Arnouts 1996, "SExtractor: Source extractor", A&AS 117, 393** — the automated source-extraction tool that ran on every modern survey for two decades. The natural pre-ML-era automation anchor.
5. **Ball & Brunner 2010, "Data mining and machine learning in astronomy", IJMPD 19, 1049 (arxiv:0906.2173)** — review of the Sloan-era ML wave (RF, SVM, NN photo-z's); the "ML existed in astro long before deep learning" reference.
6. **Baron 2019, "Machine learning in astronomy: a practical overview" (arxiv:1904.07248)** — clean bridge between Sloan-era ML and the deep-learning era; useful as a single "where things stood ~2019" anchor.

*Figure suggestion for the slide:* one composite image: Gauss's least-squares geometry → HR diagram → SExtractor segmentation map → modern CNN feature map. Each as one column.

---

## Block 2 — Recent ML applications

### 2a. Data reduction / denoising / signal cleaning

1. ✓ **Zevin et al. 2017, "Gravity Spy: integrating advanced LIGO detector characterization, ML and citizen science"** — arxiv:1611.04596. CNN + Zooniverse labels for LIGO glitch classification. Sets up the "instrumental/environmental transients mask or mimic GW signals" framing. **Where to mine figures:** glitch morphology gallery (Fig. 1) and confusion matrix.

2. ✓ **Wu et al. 2024, "Advancing the Gravity Spy classifier for LIGO O4"** — arxiv:2401.12913. Multi-time-window fusion with attention. Good "what's actually deployed in O4" update.

3. ✓ **IceCube Coll. (Abbasi et al.) 2021, "A CNN with hexagonal kernels for cascade reconstruction"** — JINST 16 (arxiv:2101.11589). 2–3 orders of magnitude speedup vs likelihood reconstruction, tested on experimental data. Loud, exactly the "ML on real instrument data" example the plan asked for. **Figure:** energy/direction residual histograms vs likelihood baseline.

4. ✓ **IceCube DeepCore 2D-CNN 2023** — arxiv:2307.16373. Exploits time and depth translational symmetry for muon-neutrino flavour ID and inelasticity; outperforms conventional likelihood reconstruction.

5. ◇ **CMB component separation with NNs (Petroff et al. 2020, "Full-sky CMB component separation with deep learning")** — known but unverified by the scan. Background-knowledge entry; verify before using.

*Star picks for the slide:* Gravity Spy (loud + visually clear) and IceCube hexagonal-kernel CNN (loud + clean speedup number).

### 2b. Classification on real instrument data

1. ✓ **Zhu et al. 2023, "Classifying unassociated Fermi-LAT sources with ML"** — arxiv:2311.03678. Voting ensemble; 91% balanced accuracy at high Galactic latitude (1,037 AGN + 88 PSR candidates); 81% at low latitude. Clean, number-rich. **Figure:** the latitude-split candidate-yield bar plots.

2. ✓ **Walmsley et al. 2020, "Galaxy Zoo: probabilistic morphology through Bayesian CNNs"** — MNRAS (arxiv:1905.07424). Combines Bayesian CNN with a generative model of volunteer responses. Cleanest example of "ML on real survey data, with calibrated uncertainties." **Figure:** posterior morphology distributions vs vote fractions.

3. ◇ **Möller & de Boissière 2020, "SuperNNova: an open-source framework for Bayesian, neural network-based supernova classification"** — known reference; not verified by scan. Use as supporting example.

4. ◇ **Carrasco-Davis et al. 2019 (ALeRCE) or Förster et al. 2021** — alert-broker classifier for ZTF; supporting example for time-domain transients on real data.

5. ✗ **Radio transient classifiers (FAST FRB CNN pipelines, MeerKAT TRAPUM)** — TODO: verify a specific paper before the slide. Likely candidates: Connor & van Leeuwen 2018, Agarwal et al. 2020 (FETCH).

*Star pick:* Walmsley 2020 (probabilistic and Galaxy-Zoo-branded, the audience-friendly anchor); Zhu 2023 as the "γ-rays count too" example.

### 2c. Anomaly / new-physics search ✗ TODO (scan returned nothing)

Candidates from background knowledge, all need verification:

1. **Collins, Howe, Nachman 2018, "Anomaly detection for resonant new physics with machine learning" (CWoLa hunting)** — LHC anchor.
2. **Nachman & Shih 2020, "Anomaly detection with density estimation" (ANODE)** — flow-based anomaly detection at colliders.
3. **Villar et al. 2021 / 2023, self-supervised learning for transients** — astro analogue of the LHC anomaly-hunt programme.
4. **Pruzhinskaya et al. 2019, "Anomaly detection in the Open Supernova Catalog"** — early astro example.
5. **LSST AGN-anomaly programme (Lochner & Bassett 2021, "Astronomaly")** — anomaly-detection toolkit on survey data.

### 2d. Fast surrogates / emulators

1. ◇ **Spurio Mancini, Piras et al. 2021/2022, "CosmoPower: emulating cosmological power spectra for ML-accelerated Bayesian inference"** — arxiv:2106.03846. The de-facto cosmology emulator in current MCMC pipelines. **Slide use:** lead example for block 3a. **Figure:** P(k) emulator residuals + speedup factor.

2. ◇ **Villaescusa-Navarro et al. 2021, "The CAMELS project"** — arxiv:2010.00619 (companion paper JINST/PASP series, see also iopscience.iop.org/article/10.3847/1538-4365/ac5ab0). Thousands of N-body+hydro sims as a training set for cosmology emulators and SBI; the canonical "simulations-as-data" infrastructure paper.

3. ◇ **Kacprzak et al. 2023, "CosmoGridV1"** or **Villaescusa-Navarro et al. 2023 "CAMELS-SAM"** — supporting examples; verify.

4. ◇ **arxiv:2510.26316** — surfaced by the scan in this angle; URL recorded but topic not pinned, **verify before citing**.

5. ◇ **arxiv:2203.08434** — same caveat.

6. ✗ **GW waveform surrogates**: NRSur7dq4 (Varma et al. 2019) is the natural anchor; surfaced as background knowledge but not by the scan. TODO.

7. ✗ **CORSIKA / air-shower surrogates**: e.g. Bourilkov 2020 "ML and the physical sciences" review, or specific Cherenkov-telescope NN reconstructions (CTAO). TODO.

### 2e. GW landmark works (the loud examples)

1. ✓ **Gabbard, Williams, Hayes, Messenger 2018, "Matching matched filtering with deep learning for GW search"** — PRL (arxiv:1712.06041). Canonical early CNN-vs-matched-filter demonstration. ROC-equivalent on the same simulated datasets. **Caveat for the slide:** equivalence is on simulated data; generalisation to real noise has been critiqued in follow-up work. **Figure:** ROC curve overlay.

2. ✓ **Nousi et al. 2022/2024, "AresGW"** — arxiv:2211.01520 and arxiv:2407.07820. 54-layer ResNet with Deep Adaptive Input Normalization, dynamic augmentation, curriculum learning. Detects non-aligned-spin BBH in real LIGO O3a noise; 2024 paper reports new ML-only candidate events. **Caveat:** parameter-space-limited (7–50 M☉, non-aligned spin). The "modern, on real data" GW detector net.

3. ✓ **Dax et al. 2021, "Real-time gravitational-wave science with neural posterior estimation" (DINGO)** — PRL 127, 241103 (arxiv:2106.12594). O(day) → ~1 minute per event; matches standard samplers on 8 GWTC-1 events. **Star pick for slide 11. Figure:** posterior corner overlay vs LALInference.

4. ✓ **Dax et al. 2022, "Neural importance sampling for rapid and reliable GW inference" (DINGO-IS)** — arxiv:2210.05686. Adds importance sampling on top of NPE; unbiased posteriors with a built-in sample-efficiency failure diagnostic. ~10% median sample efficiency on 42 BBH events, "two orders of magnitude better than standard samplers." **Figure:** efficiency histogram.

5. ✓ **Buchli, Tracey et al. (DeepMind + LIGO Instrument Team) 2025, "Deep Loop Shaping"** — Science 389, 6764 (arxiv:2509.14016, DOI 10.1126/science.adw1291). RL with frequency-domain rewards for mirror-suspension control at LIGO Livingston. **>30× control-noise reduction** in 10–30 Hz, up to 100× in sub-bands. **This is the genuine DeepMind/LIGO joint paper the brief asked about.** Loud, recent, on real detector. **Star pick for slide 12. Figure:** noise spectrum before/after.

### 2f. Foundation / large models touching astro

The brief asked specifically about Mistral-in-astro. **The scan found no Mistral-authored or Mistral-applied-to-astro paper of the kind the plan implied.** Frame the slide accordingly: foundation models in astro exist (AstroCLIP, AstroLLaMA), but no headline result from Mistral itself to my knowledge as of 2026-06.

1. ◇ **Nguyen et al. 2023, "AstroLLaMA: towards specialized foundation models for astronomy"** — arxiv:2309.06126. Domain-adapted LLaMA-2 on arxiv astro-ph abstracts. The natural "LLMs read astronomy" anchor.

2. ◇ **Parker et al. 2024, "AstroCLIP: a cross-modal foundation model for galaxies"** — arxiv:2310.03024. Contrastive image-spectrum embedding, transferable to photo-z, morphology, redshift. **Star pick for slide 13. Figure:** image-spectrum joint embedding scatter.

3. ◇ **arxiv:2405.14930** ("Multimodal Universe" or similar — verify) and ◇ **arxiv:2409.20252** ("AstroPT" or a follow-up) — surfaced by the scan, verify.

4. ◇ **arxiv:2510.05016** — recent foundation-model-in-astro paper, surfaced by the scan, verify title and angle.

5. ✗ **LLM-as-research-assistant papers in astro (e.g., Wong et al. 2023 "Can ChatGPT pass an astro exam", Ciucă et al. 2023, AstroPilot)** — supporting examples, none verified.

### 2g. End-to-end inference on real data beyond GW ✗ TODO

The scan did not specifically cover this; verify each before slide-time.

1. **Hezaveh, Levasseur, Marshall 2017, "Fast automated analysis of strong gravitational lenses with CNNs"** — early landmark on real HST cutouts.
2. **Wagner-Carena et al. 2022 / 2023, "A hierarchical strong-lensing analysis with SBI"** — modern SBI on real HST lenses.
3. **Saxena et al. 2023 / Zhao et al. 2024, 21cm SBI on HERA Phase-1 data** — verify which is the right anchor.
4. **Hortúa et al. 2020 / Petroff et al. 2020, "Neural cosmological parameter inference from CMB maps"** — Planck/ACT-adjacent.
5. **Modi et al. 2023, "Sensitivity-aware amortised Bayesian inference at SDSS-III"** — supporting example.

---

## Block 3 — Simulation-based inference (the segue)

### 3a. Likelihood-based inference "on steroids"

1. ◇ **Spurio Mancini, Piras et al. 2021/2022, CosmoPower** — already listed in 2d. Here used as the "neural emulator inside a standard MCMC" anchor.

2. ✗ **Auld, Bridges, Hobson, Gull 2007, "CosmoNet"** — the original neural emulator for CMB likelihoods. Background-knowledge anchor for "this idea is old."

3. ✗ **Heitmann et al. 2009 onwards, "Coyote Universe / Mira-Titan / Aemulus"** — emulator-based likelihoods for large-scale structure. Background-knowledge anchor.

4. ✗ **Modi et al. 2023, Cabezas et al. 2023, normalising-flow likelihood surrogates inside HMC** — supporting examples.

### 3b. SBI methods proper (the "ABC on steroids" anchors)

1. ◇ **Cranmer, Brehmer, Louppe 2020, "The frontier of simulation-based inference"** — PNAS, arxiv:1911.01429. The community-standard review. **Star pick to anchor the SBI segue slide.** **Figure:** the now-iconic taxonomy diagram (NPE/NLE/NRE).

2. ◇ **Greenberg, Nonnenmacher, Macke 2019, "Automatic posterior transformation for likelihood-free inference" (SNPE-C)** — proceedings.mlr.press/v97/greenberg19a. The flow-based amortised-NPE workhorse.

3. ◇ **Papamakarios, Sterratt, Murray 2019, "Sequential neural likelihood" (SNL)** — arxiv:1903.04057. NLE counterpart.

4. ◇ **Hermans, Begy, Louppe 2020, "Likelihood-free MCMC with amortised approximate ratio estimators" (NRE / AALR)** — verify ID (commonly arxiv:1903.04057 is SNL; NRE methods paper is Hermans et al. 2020 ICML, arxiv:1903.04057's sibling). Listed in scan as arxiv:2205.09126 — likely the **balanced NRE / sbi-bench** paper instead. Verify.

5. ◇ **Tejero-Cantero et al. 2020, "sbi: a toolkit for simulation-based inference"** — JOSS. The library reference. The scan also surfaced **academic.oup.com/rasti/article/3/1/724/7888995** which appears to be a related SBI library or benchmarks paper; verify.

6. ✗ **Papamakarios & Murray 2016, "Fast ε-free inference of simulation models with Bayesian conditional density estimation"** — NeurIPS. The historical anchor before the SNPE/SNL/SNRE naming consolidated. Worth one line on the segue slide.

### 3c. SBI applied to APP/GW specifically

1. ✓ **DINGO + DINGO-IS (Dax 2021, 2022)** — already in 2e; here it's the bridge from "ML detects GWs" to "ML *infers parameters* from GWs," which is exactly the framing the plan asks for in this block.

2. ◇ **Cole et al. 2022, "Fast and credible likelihood-free cosmology with TMNRE" (swyft)** — recent SBI-applied-to-cosmology anchor. Verify exact arxiv ID.

3. ◇ **Anau Montel, Coogan, Correa, Karchev, Weniger 2022/2023, TMNRE for strong lensing and dark substructure** — self-cite candidate; verify which paper is the right one for the slide.

4. ✗ **Saxena et al. 2023, SBI for 21cm cosmology on HERA-like data** — TODO.

5. ✗ **Modi et al. 2023, "Sensitivity-aware amortised inference" for galaxy-clustering** — TODO.

6. ✗ **Field-level cosmology with SBI (Dai & Seljak 2022, Lemos et al. 2023)** — supporting examples.

---

## Open questions for the next iteration

1. **History block** is uncurated. Want me to do a focused second pass just on slide 1?
2. **Mistral-in-astro**: confirm the slide will say "no such paper as of June 2026, here are the actual foundation-model-in-astro anchors" rather than chase a nonexistent reference.
3. **AresGW caveat**: the parameter-space limitation (7–50 M☉, non-aligned spin) should appear on the slide, not just in this longlist. Worth a slide footnote.
4. **Gabbard 2018 caveat**: ROC-equivalence was on simulated data; later work has critiqued generalisation. Frame on the slide as "early demonstration, real-noise performance came later (AresGW)."
5. **Block 2g** (end-to-end inference on real data beyond GW) is the thinnest block of the lecture and needs a dedicated second-pass search if it gets its own slide.
6. **SBI methods slide**: do we want a method-zoo slide (NPE/NLE/NRE side by side) or a single anchor (Cranmer-Brehmer-Louppe taxonomy figure)? The latter is cleaner for a 45-min overview; the former matches Lec 3's needs better. Decide before drafting the slide.
