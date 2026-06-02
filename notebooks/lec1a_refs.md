# Lec 1a — Annotated reference longlist

Target: 45 min, ~15 slides, 1–2 figures per slide, ≥1 reference per slide. Time window: 2018–2026 emphasis, with pre-2018 anchors only for the history block.

**Verification key:**
- ✓ = primary source verified 3-0 by the deep-research workflow (claim, authors, headline numbers checked against arxiv abstract).
- ◇ = surfaced by the research scan, primary URL in hand, but **not** adversarially verified. Skim before quoting numbers.
- ✗ = no source found by the scan; entry below is from background knowledge or marked TODO. Verify before using.

The plan asked us to be explicit about loud-but-unconfirmed examples. The DeepMind/LIGO paper is real and verified ✓. A Mistral-in-astro paper of the kind the brief described **was not found** by the scan and should be presented as "no such paper to my knowledge as of 2026-06" rather than fabricated.

---

## Star picks per slide

A first cut at one anchor paper per slide, to be confirmed once we lock the slide-level outline. Slide numbers are tentative and assume ~15 slides.

| # | Slide topic                                       | Star pick                                              | Status |
|---|---------------------------------------------------|--------------------------------------------------------|--------|
| 1 | Title + history arc                                | SExtractor (Bertin & Arnouts 1996) + Gauss/Ceres anchor| ✗ TODO |
| 2 | The ML-in-physics wave (framing)                   | Cranmer, Brehmer, Louppe 2020 "Frontier of SBI"        | ◇      |
| 3 | Data reduction: GW glitch classification           | Gravity Spy (Zevin 2017 + Wu 2024)                     | ✓      |
| 4 | Data reduction: neutrino event reconstruction      | IceCube cascade-CNN (Abbasi 2021)                      | ✓      |
| 5 | Classification on real data: γ-ray sources         | Fermi-LAT ensemble (Zhu 2023)                          | ✓      |
| 6 | Classification on real data: galaxy morphology     | Galaxy Zoo Bayesian CNN (Walmsley 2020)                | ✓      |
| 7 | Anomaly / new-physics search                       | Self-supervised transients (Villar 2023, ◇) or LHC ADC | ✗ TODO |
| 8 | Fast surrogates: cosmology emulators               | CosmoPower (Spurio Mancini 2021)                       | ◇      |
| 9 | Fast surrogates: GW waveforms or N-body            | CAMELS (Villaescusa-Navarro 2021)                      | ◇      |
| 10| GW detection nets                                  | Gabbard 2018 (historical) + AresGW 2024 (current)      | ✓      |
| 11| GW parameter inference                             | DINGO (Dax 2021) + DINGO-IS (Dax 2022)                 | ✓      |
| 12| LIGO interferometer control                        | Deep Loop Shaping (Buchli et al., Science 2025)        | ✓      |
| 13| Foundation models touching astro                   | AstroCLIP (Parker 2024) + AstroLLaMA (Nguyen 2023)     | ◇      |
| 14| Likelihood-based inference "on steroids"           | CosmoPower-HMC pipelines                               | ◇      |
| 15| SBI ("ABC on steroids") + segue to Lec 1b          | Cranmer, Brehmer, Louppe 2020 + sbi library            | ◇      |

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
