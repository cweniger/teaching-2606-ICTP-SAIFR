# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2606-ICTP-SAIFR/blob/main/notebooks/s1_app_gw.ipynb)
#
# # Session 1 — APP example: gravitational-wave chirp
#
# **Session 1, Block 3 (~45 min). Runs on a laptop CPU.**
#
# Take the Gaussian-head NPE machinery you built in
# [s1_pytorch_and_npe](./s1_pytorch_and_npe.ipynb) and point it at an
# astroparticle-flavoured simulator: an equal-mass binary black-hole
# inspiral-merger-ringdown (IMR) waveform in coloured advanced-LIGO
# noise. Two parameters,
#
# - **chirp mass** $\mathcal{M}\in[10,60]\,M_\odot$,
# - **luminosity distance** $d_L\in[100, 2000]$ Mpc,
#
# everything else (mass ratio fixed equal, optimal sky/orientation,
# coalescence at segment centre) is baked into the simulator.
#
# **What you build:** a 4-second strain time series at 2048 Hz → a 2-D
# hand-built matched-filter summary → the **same** `GaussianHead` and
# `gaussian_nll` from Block 2, only with `in_dim=2, out_dim=4`. The
# residual chirp-mass / distance correlation that the diagonal Gaussian
# head visibly misses becomes your motivation for normalising flows in
# Session 2.

# %%
# !pip install -q --upgrade --force-reinstall --no-deps git+https://github.com/cweniger/teaching-2606-ICTP-SAIFR.git  # noqa: E501

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from samma_sbi.simulators import GWChirp

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# %% [markdown]
# ---
#
# ## 1 — Inside the simulator
#
# Under the hood, `GWChirp` is the Ajith et al. 2007 PhenomA waveform
# evaluated at equal mass ratio, added to coloured Gaussian noise drawn
# from an analytic aLIGO-design PSD. You do not need any of those words
# to use it; the interface mirrors `BallThrow`:

# %%
sim = GWChirp()
print(f"duration  = {sim.duration} s")
print(f"f_sample  = {sim.f_sample} Hz")
print(f"n_samples = {sim.n_samples}  (= {sim.duration}s x {sim.f_sample}Hz)")
print(f"prior     : Mc in [{sim.mc_low}, {sim.mc_high}] M_sun, "
      f"dL in [{sim.dl_low}, {sim.dl_high}] Mpc")

# %% [markdown]
# **Forward simulation.** A single call gives one realisation of strain
# (signal + noise) sampled at 2048 Hz over 4 seconds.

# %%
rng = np.random.default_rng(SEED)
theta_demo = np.array([30.0, 500.0])  # Mc = 30 M_sun, dL = 500 Mpc
x_demo = sim.simulate(theta_demo, rng=rng)
print("x shape:", x_demo.shape, "  amplitude:", float(np.std(x_demo)))

# %% [markdown]
# Plot the time series, then the *whitened* time series — the signal is
# essentially invisible in raw strain (drowned in low-frequency noise)
# but pops out clearly after whitening by $1/\sqrt{S_n(f)}$.

# %%
def whiten(x, sim):
    Xf = np.fft.rfft(x)
    Xf = Xf / np.sqrt(sim.psd(sim.freqs))
    return np.fft.irfft(Xf, n=sim.n_samples)


t = np.arange(sim.n_samples) / sim.f_sample
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4), sharex=True)
ax1.plot(t, x_demo, color="C7", lw=0.4)
ax1.set_ylabel("raw strain")
ax2.plot(t, whiten(x_demo, sim), color="C0", lw=0.6)
ax2.set_xlabel("time [s]"); ax2.set_ylabel("whitened strain")
ax2.set_xlim(1.5, 2.3)  # zoom around the merger at t=2
fig.suptitle(r"$\mathcal{M}_c = 30\,M_\odot,\ d_L = 500$ Mpc")
fig.tight_layout(); plt.show()

# %% [markdown]
# The chirp leading into the merger spike at $t = 2\,$s is the
# astrophysical event you are trying to characterise.
#
# **Noiseless template, for reference.** The same waveform without
# adding noise lets you see the chirp + merger + ringdown structure
# directly.

# %%
h_fd = sim._waveform_fd(30.0, 500.0)            # frequency-domain template
h_t = np.fft.irfft(h_fd, n=sim.n_samples)        # time-domain
plt.figure(figsize=(7.5, 2.2))
plt.plot(t, h_t, color="C3", lw=0.8)
plt.xlim(1.9, 2.05); plt.xlabel("time [s]"); plt.ylabel("strain (noiseless)")
plt.title("PhenomA template — chirp, merger, ringdown")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 1.1 — The hand-built summary statistic
#
# A raw 8192-sample time series is too big to feed directly into a tiny
# MLP. We need a low-dimensional summary that retains the information
# about $(\mathcal{M}_c, d_L)$. The simulator provides one:
#
# - a **2-template matched-filter mini-bank**: two reference templates,
#   one light ($\mathcal{M}_\mathrm{c, ref} = 15\,M_\odot$) and one
#   heavy ($45\,M_\odot$), both at $d_{L,\mathrm{ref}} = 500$ Mpc;
# - `sim.summary(x)` returns the two normalised matched-filter responses
#   $(\rho_\mathrm{lo}, \rho_\mathrm{hi})$. Under pure noise each is
#   $\mathcal{N}(0, 1)$; under a true signal, the *magnitudes* track
#   $d_L^{-1}$ and the *ratio* tracks chirp mass.
#
# Let's see the summary depend on the parameters.

# %%
rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
mc_show = np.linspace(sim.mc_low, sim.mc_high, 5)
for c, mc in zip(colors, mc_show):
    pts = []
    for dl in np.linspace(sim.dl_low, sim.dl_high, 4):
        for _ in range(15):
            s = sim.summary(sim.simulate(np.array([mc, dl]), rng=rng))
            pts.append(s)
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color=c, alpha=0.6,
               label=fr"$\mathcal{{M}}_c = {mc:.0f}\,M_\odot$")
ax.axhline(0, color="0.7", lw=0.5); ax.axvline(0, color="0.7", lw=0.5)
ax.set_xlabel(r"$\rho_\mathrm{lo}$ (matched to 15 $M_\odot$ template)")
ax.set_ylabel(r"$\rho_\mathrm{hi}$ (matched to 45 $M_\odot$ template)")
ax.legend(loc="best", fontsize=8); ax.set_title("Summary statistic across the prior")
fig.tight_layout(); plt.show()

# %% [markdown]
# Read this plot. As you sweep $\mathcal{M}_c$ from light to heavy, the
# cloud of summary points rotates from
# *(high $\rho_\mathrm{lo}$, low $\rho_\mathrm{hi}$)* to
# *(low $\rho_\mathrm{lo}$, high $\rho_\mathrm{hi}$)* — the heavier
# template starts matching the signal better. As you sweep $d_L$ from
# near to far, the magnitudes shrink toward $(0, 0)$. Two scalars that
# encode the two parameters: exactly what NPE needs.

# %% [markdown]
# ---
#
# ## 2 — Training set
#
# Same recipe as Block 2 of session 1, one dimension up:
# $\theta \in \mathbb{R}^2$, $s = \text{summary}(x) \in \mathbb{R}^2$.

# %%
def simulate_dataset(sim, n_pairs, rng):
    theta = sim.sample_prior(n_pairs, rng=rng)
    x = sim.simulate(theta, rng=rng)
    s = sim.summary(x)
    return (
        torch.tensor(theta, dtype=torch.float32),
        torch.tensor(s, dtype=torch.float32),
    )


N_TRAIN, N_VAL = 8000, 800
rng = np.random.default_rng(SEED)
theta_tr, s_tr = simulate_dataset(sim, N_TRAIN, rng)
theta_va, s_va = simulate_dataset(sim, N_VAL, rng)
print("train:", theta_tr.shape, s_tr.shape)
print("val:  ", theta_va.shape, s_va.shape)

# %% [markdown]
# **Normalisation matters.** $\mathcal{M}_c \in [10, 60]$ and
# $d_L \in [100, 2000]$ are on very different scales. We z-score both
# the parameter and summary axes before training; the network output
# means/log-variances live on the normalised scale and we map back at
# evaluation time.

# %%
theta_mu, theta_sd = theta_tr.mean(0), theta_tr.std(0)
s_mu, s_sd = s_tr.mean(0), s_tr.std(0)
theta_tr_n = (theta_tr - theta_mu) / theta_sd
theta_va_n = (theta_va - theta_mu) / theta_sd
s_tr_n = (s_tr - s_mu) / s_sd
s_va_n = (s_va - s_mu) / s_sd
print("theta_mu:", theta_mu.numpy(), " theta_sd:", theta_sd.numpy())
print("s_mu:    ", s_mu.numpy(),     " s_sd:    ", s_sd.numpy())

# %% [markdown]
# ---
#
# ## 3 — Gaussian-head NPE (same code as Block 2, wider inputs/outputs)
#
# Same model class you wrote in `s1_pytorch_and_npe`, only with two
# differences:
#
# - input dimension is 2 (the summary statistic),
# - output dimension is 2 means + 2 log-variances → 4 numbers, treated
#   as a **diagonal-covariance** 2-D Gaussian.
#
# Diagonal means the network can capture independent uncertainty in
# $\mathcal{M}_c$ and $d_L$, but cannot represent their correlation.
# That is a deliberate limitation; see the final section.

# %%
class GaussianHead2D(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden, out_dim)
        self.head_logvar = nn.Linear(hidden, out_dim)

    def forward(self, s):
        h = self.trunk(s)
        return self.head_mu(h), self.head_logvar(h)


def gaussian_nll_2d(theta, mu, log_var):
    # Diagonal Gaussian NLL, summed over the parameter axis, averaged over batch.
    var = torch.exp(log_var)
    return 0.5 * (((theta - mu) ** 2) / var + log_var).sum(dim=-1).mean()


# %% [markdown]
# Same five-step training loop, plus a held-out validation pass.

# %%
def train(s_tr, theta_tr, s_va, theta_va,
          n_epochs=120, batch_size=256, lr=1e-3, hidden=64, seed=SEED):
    torch.manual_seed(seed)
    model = GaussianHead2D(in_dim=s_tr.shape[1], hidden=hidden,
                           out_dim=theta_tr.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    n = s_tr.shape[0]
    tr_curve, va_curve = [], []
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        ep = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            mu, log_var = model(s_tr[idx])
            loss = gaussian_nll_2d(theta_tr[idx], mu, log_var)
            loss.backward(); opt.step()
            ep += loss.item() * idx.numel()
        tr_curve.append(ep / n)
        with torch.no_grad():
            mu_v, lv_v = model(s_va)
            va_curve.append(gaussian_nll_2d(theta_va, mu_v, lv_v).item())
    return model, np.array(tr_curve), np.array(va_curve)


model, tr_curve, va_curve = train(s_tr_n, theta_tr_n, s_va_n, theta_va_n)

plt.figure(figsize=(5, 3))
plt.plot(tr_curve, label="train"); plt.plot(va_curve, label="val")
plt.xlabel("epoch"); plt.ylabel("Gaussian NLL"); plt.legend()
plt.title("Gaussian-head NPE — training"); plt.tight_layout(); plt.show()

# %% [markdown]
# ---
#
# ## 4 — Evaluate against a grid reference posterior
#
# Pick a true $(\mathcal{M}_c, d_L)$, simulate an observation, evaluate
# both the NPE posterior and the brute-force matched-filter
# reference posterior, and overlay.

# %%
theta_true = np.array([35.0, 600.0])
x_obs = sim.simulate(theta_true, rng=np.random.default_rng(123))
s_obs = sim.summary(x_obs)

# NPE posterior on a (Mc, dL) grid: evaluate q_phi at s_obs once,
# then write down the diagonal Gaussian density at every grid point.
s_obs_n = (torch.tensor(s_obs, dtype=torch.float32) - s_mu) / s_sd
with torch.no_grad():
    mu_n, log_var_n = model(s_obs_n.unsqueeze(0))
mu_n = mu_n.squeeze(0); sigma_n = torch.exp(0.5 * log_var_n.squeeze(0))
# back to physical scale
mu_phys = (mu_n * theta_sd + theta_mu).numpy()
sigma_phys = (sigma_n * theta_sd).numpy()
print(f"NPE: Mc = {mu_phys[0]:.2f} +- {sigma_phys[0]:.2f},  "
      f"dL = {mu_phys[1]:.1f} +- {sigma_phys[1]:.1f}")

# Reference posterior (matched-filter likelihood on a grid). Takes a
# few seconds.
mc_g, dl_g, post_ref = sim.true_posterior(x_obs, n_mc=70, n_dl=70)

# NPE diagonal-Gaussian density on the same grid:
MC, DL = np.meshgrid(mc_g, dl_g, indexing="ij")
post_npe = (1 / (2 * np.pi * sigma_phys[0] * sigma_phys[1])) * np.exp(
    -0.5 * (((MC - mu_phys[0]) / sigma_phys[0]) ** 2
            + ((DL - mu_phys[1]) / sigma_phys[1]) ** 2)
)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.contour(MC, DL, post_ref, levels=8, colors="C7", linewidths=0.8)
ax.contour(MC, DL, post_npe, levels=8, colors="C0", linewidths=1.2)
ax.plot(theta_true[0], theta_true[1], "C3*", ms=12,
        label=fr"$\theta_\mathrm{{true}}=({theta_true[0]:.0f},\,{theta_true[1]:.0f})$")
ax.set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$")
ax.set_ylabel(r"$d_L$ [Mpc]")
ax.set_title("grid reference (grey) vs. Gaussian-head NPE (blue)")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); plt.show()

# %% [markdown]
# ### Exercise — amortisation
#
# Repeat the above plot for three different $(\mathcal{M}_c, d_L)$
# truths *without retraining* — the same network produces every
# posterior. Confirm visually that the NPE Gaussian sits on top of the
# reference peak in each case.

# %%
# TODO — your code here.


# %%
# @title Reference solution { display-mode: "form" }
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, (mc_t, dl_t) in zip(axes, [(15.0, 300.0), (40.0, 1000.0), (55.0, 1600.0)]):
    x0 = sim.simulate(np.array([mc_t, dl_t]), rng=np.random.default_rng(int(mc_t * dl_t)))
    s0 = sim.summary(x0)
    s0_n = (torch.tensor(s0, dtype=torch.float32) - s_mu) / s_sd
    with torch.no_grad():
        mu_n, lv_n = model(s0_n.unsqueeze(0))
    mu_p = (mu_n.squeeze(0) * theta_sd + theta_mu).numpy()
    sg_p = (torch.exp(0.5 * lv_n.squeeze(0)) * theta_sd).numpy()
    mc_g0, dl_g0, post_ref0 = sim.true_posterior(x0, n_mc=60, n_dl=60)
    MC0, DL0 = np.meshgrid(mc_g0, dl_g0, indexing="ij")
    post_npe0 = (1 / (2 * np.pi * sg_p[0] * sg_p[1])) * np.exp(
        -0.5 * (((MC0 - mu_p[0]) / sg_p[0]) ** 2
                + ((DL0 - mu_p[1]) / sg_p[1]) ** 2))
    ax.contour(MC0, DL0, post_ref0, levels=6, colors="C7", linewidths=0.8)
    ax.contour(MC0, DL0, post_npe0, levels=6, colors="C0", linewidths=1.2)
    ax.plot(mc_t, dl_t, "C3*", ms=10)
    ax.set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$")
    ax.set_title(f"truth = ({mc_t:.0f}, {dl_t:.0f})")
axes[0].set_ylabel(r"$d_L$ [Mpc]")
fig.tight_layout(); plt.show()

# %% [markdown]
# ---
#
# ## 5 — Where the Gaussian head will hurt
#
# Look closely at the reference contours in the previous plot. The
# grid posterior is **banana-shaped**: a louder, lighter chirp can look
# similar to a quieter, heavier one because both depend on the same
# matched-filter amplitude. There is a real $\mathcal{M}_c$–$d_L$
# degeneracy in the data.
#
# Your diagonal-Gaussian $q_\phi$ cannot represent that correlation;
# it can only stretch axes-aligned ellipses. So at moderate-to-low SNR
# it will visibly *over-cover* one direction and *under-cover* another.
# That is not a bug in NPE — it is the inevitable consequence of
# choosing too restrictive a density family.
#
# **This is the failure that motivates Lecture 3 and Session 2.** Pick
# a low-SNR truth (large $d_L$, light $\mathcal{M}_c$) and look at the
# tilt of the reference posterior compared with your NPE ellipse:

# %%
theta_low = np.array([15.0, 1700.0])
x_low = sim.simulate(theta_low, rng=np.random.default_rng(42))
s_low = sim.summary(x_low)
s_low_n = (torch.tensor(s_low, dtype=torch.float32) - s_mu) / s_sd
with torch.no_grad():
    mu_l, lv_l = model(s_low_n.unsqueeze(0))
mu_lp = (mu_l.squeeze(0) * theta_sd + theta_mu).numpy()
sg_lp = (torch.exp(0.5 * lv_l.squeeze(0)) * theta_sd).numpy()
mc_g2, dl_g2, post_ref2 = sim.true_posterior(x_low, n_mc=80, n_dl=80)
MC2, DL2 = np.meshgrid(mc_g2, dl_g2, indexing="ij")
post_npe2 = (1 / (2 * np.pi * sg_lp[0] * sg_lp[1])) * np.exp(
    -0.5 * (((MC2 - mu_lp[0]) / sg_lp[0]) ** 2
            + ((DL2 - mu_lp[1]) / sg_lp[1]) ** 2))

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.contour(MC2, DL2, post_ref2, levels=10, colors="C7", linewidths=0.8)
ax.contour(MC2, DL2, post_npe2, levels=10, colors="C0", linewidths=1.2)
ax.plot(*theta_low, "C3*", ms=12, label="truth")
ax.set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$"); ax.set_ylabel(r"$d_L$ [Mpc]")
ax.set_title("Low-SNR truth — banana posterior vs axis-aligned ellipse")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); plt.show()

# %% [markdown]
# Hold on to this notebook. In Session 2 we replace `GaussianHead2D`
# with a normalising flow and watch the blue contour bend into the
# grey banana.
