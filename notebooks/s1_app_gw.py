# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2606-ICTP-SAIFR/blob/main/notebooks/s1_app_gw.ipynb)
#
# # Session 1B — A real(istic) inference problem: a gravitational-wave chirp
#
# **Hands-on session 1B (after Lecture 2). Runs on a laptop CPU. ~60 min.**
#
# Session 1A ended with a problem left open. The Gaussian *band* you built,
# `q_φ(θ|x) = N(μ_θ(x), σ_θ²)`, predicted the mean beautifully but used a
# **single shared width** `σ_θ`, so it was too wide where the posterior
# is narrow and too narrow where the posterior is wide. The width should
# depend on `x`.
#
# Today we (1) fix that with a width that depends on `x`
# (**heteroscedastic**), (2) point the same machinery at a real
# gravitational-wave simulator with **two** parameters, and (3) upgrade
# the head to a full **2-D Gaussian with a learned correlation** so it
# can capture the tilted, banana-shaped posterior that two correlated
# parameters produce. We close with a quick preview of a normalising
# **flow** from the `sbi` library, the tool the next session is built on.
#
# **What's genuinely new here:**
#
# - the observation is a *time series* (8192 numbers), so we cannot feed
#   it raw to a tiny MLP: we need a **summary statistic**, and you build
#   one by hand from matched filtering;
# - the posterior lives in *2-D* and its parameters are *correlated*.

# %%
# Install the samma_sbi course package from GitHub.
!pip install -q --upgrade --force-reinstall --no-deps git+https://github.com/cweniger/teaching-2606-ICTP-SAIFR.git  # noqa: E501

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange   # progress bars; tqdm.auto picks the right style for Colab

from samma_sbi.simulators import BallThrow, GWChirp
from samma_sbi.viz import credible_levels   # 68% / 95% contour levels from a density grid

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# %% [markdown]
# ---
#
# ## 1 — Close the Session-1A gap: let the width depend on `x`
#
# Before the gravitational waves, a two-minute fix of the band. The only
# change from Session 1A is that the log-variance is now a **second
# output of the network**, `log σ²_θ(x)`, instead of one shared
# `nn.Parameter`. That is the whole idea of a **heteroscedastic** head.
# We show it on the same ball-throw, at the same two observations where
# the shared-width band failed.

# %%
def gaussian_nll(theta, mu, log_var):
    """1-D Gaussian NLL, averaged over the batch (from Session 1A)."""
    return 0.5 * (((theta - mu) ** 2) / torch.exp(log_var) + log_var).mean()


class HeteroBand1D(nn.Module):
    """Gaussian band whose width now depends on x: mu_θ(x) AND log σ²_θ(x)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden, 1)
        self.head_logvar = nn.Linear(hidden, 1)   # <-- the new part

    def forward(self, x):
        h = self.trunk(x)
        return self.head_mu(h), self.head_logvar(h)


# Train on the ball-throw (same setup as Session 1A, prior (0.05, pi/4)).
N_BALLS = 10
sim_bt = BallThrow(prior_low=0.05, prior_high=np.pi / 4)
rng = np.random.default_rng(SEED)
theta_bt = sim_bt.sample_prior(4000, rng=rng)
x_bt = sim_bt.simulate_summary(theta_bt, n_balls=N_BALLS, rng=rng)
theta_bt = torch.tensor(theta_bt, dtype=torch.float32).unsqueeze(1)
x_bt = torch.tensor(x_bt, dtype=torch.float32).unsqueeze(1)

model_bt = HeteroBand1D()
opt = optim.Adam(model_bt.parameters(), lr=1e-3)
bar = trange(120, desc="train band (ball-throw)", leave=False)
for _ in bar:
    perm = torch.randperm(x_bt.shape[0])
    last = 0.0
    for i in range(0, x_bt.shape[0], 256):
        idx = perm[i:i + 256]
        opt.zero_grad()
        loss = gaussian_nll(theta_bt[idx], *model_bt(x_bt[idx]))
        loss.backward(); opt.step()
        last = loss.item()
    bar.set_postfix(loss=f"{last:.3f}")

theta_grid_bt = np.linspace(sim_bt.prior_low, sim_bt.prior_high, 600)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, theta_t in zip(axes, [0.15, 0.70]):
    x_obs = float(sim_bt.simulate_summary(np.array([theta_t]), n_balls=N_BALLS,
                                          rng=np.random.default_rng(int(1000 * theta_t)))[0])
    with torch.no_grad():
        mu, lv = model_bt(torch.tensor([[x_obs]], dtype=torch.float32))
    mu = mu.item(); sg = float(torch.exp(0.5 * lv).item())
    q = (1 / (sg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((theta_grid_bt - mu) / sg) ** 2)
    _, p_true = sim_bt.true_posterior(x_obs, n_balls=N_BALLS, theta_grid=theta_grid_bt)
    ax.plot(theta_grid_bt, p_true, "k-", lw=1.5, label="exact posterior")
    ax.plot(theta_grid_bt, q, "C0--", lw=1.5, label="heteroscedastic band")
    ax.axvline(theta_t, color="C3", lw=1)
    ax.set_xlabel(r"$\theta$"); ax.set_title(fr"$\theta_{{\rm true}} = {theta_t}$")
    ax.legend(fontsize=8)
axes[0].set_ylabel("density")
fig.suptitle("width now depends on x: narrow where it should be, wide where it should be")
fig.tight_layout(); plt.show()

# %% [markdown]
# The band is now **narrow** near `θ = 0.15` and **wide** near
# `θ = 0.70`, matching the exact posterior at both. Letting `σ` be a
# network output, not a constant, fixed the Session-1A shortcoming. Keep
# this idea; we use it again immediately, in 2-D.
#
# ---
#
# ## 2 — The gravitational-wave simulator
#
# `GWChirp` forward-models the strain from an equal-mass binary
# black-hole inspiral-merger-ringdown in coloured advanced-LIGO noise.
# Two parameters are free:
#
# - **chirp mass** `Mc` (sets how the signal sweeps up in frequency),
# - **luminosity distance** `dL` (sets how loud it is).
#
# Everything else is fixed inside the simulator. The interface mirrors
# `BallThrow`.

# %%
sim = GWChirp()
print(f"observation length : {sim.n_samples} samples "
      f"({sim.duration}s x {sim.f_sample:.0f} Hz)")
print(f"prior : Mc in [{sim.mc_low}, {sim.mc_high}] Msun, "
      f"dL in [{sim.dl_low}, {sim.dl_high}] Mpc")

# %% [markdown]
# **One observation.** Strain (signal + noise) over 4 seconds. The raw
# strain is dominated by low-frequency noise and the signal is invisible;
# after **whitening** (dividing each frequency by the noise amplitude
# `√S_n(f)`, so every frequency contributes on an equal footing) the
# chirp leading into the merger at `t = 2 s` pops out.

# %%
def whiten(x, sim):
    Xf = np.fft.rfft(x) / np.sqrt(sim.psd(sim.freqs))
    return np.fft.irfft(Xf, n=sim.n_samples)


theta_true = np.array([30.0, 600.0])   # Mc = 30 Msun, dL = 600 Mpc
x_demo = sim.simulate(theta_true, rng=np.random.default_rng(1))
t = np.arange(sim.n_samples) / sim.f_sample

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4), sharex=True)
ax1.plot(t, x_demo, color="C7", lw=0.4); ax1.set_ylabel("raw strain")
ax2.plot(t, whiten(x_demo, sim), color="C0", lw=0.6)
ax2.set_xlabel("time [s]"); ax2.set_ylabel("whitened"); ax2.set_xlim(1.6, 2.2)
fig.suptitle(r"$M_c = 30\,M_\odot,\ d_L = 600$ Mpc")
fig.tight_layout(); plt.show()

# %% [markdown]
# **How the two parameters change the signal.** Below are noiseless
# whitened templates: lighter `Mc` chirps for longer and reaches higher
# frequency; nearer `dL` (smaller) is louder. Try changing the values.

# %%
plt.figure(figsize=(7.5, 2.6))
for mc, dl, c in [(22, 600, "C0"), (40, 600, "C1"), (30, 350, "C2")]:
    h = np.fft.irfft(sim._waveform_fd(mc, dl), n=sim.n_samples)
    plt.plot(t, whiten(h, sim), color=c, lw=0.9, label=fr"$M_c={mc},\ d_L={dl}$")
plt.xlim(1.85, 2.02); plt.xlabel("time [s]"); plt.ylabel("whitened strain")
plt.legend(fontsize=8); plt.title("noiseless templates"); plt.tight_layout(); plt.show()

# %% [markdown]
# **Signal loudness (SNR) across the prior.** The same `Mc` at the near
# edge of the prior is a loud, easy detection; at the far edge it is
# marginal. This range is what makes the inference interesting.

# %%
for mc in [25, 32, 40]:
    snrs = [f"dL={dl}: {sim.optimal_snr(mc, dl):4.1f}" for dl in [300, 600, 1200]]
    print(f"Mc={mc:>3} Msun   " + "   ".join(snrs))

# %% [markdown]
# ---
#
# ## 3 — From 8192 numbers to 2: a hand-built summary
#
# We cannot feed 8192-sample strain into a tiny MLP. We need a **summary
# statistic**: a few numbers that keep the information about
# `(Mc, dL)`. The classical gravitational-wave tool is the **matched
# filter**: correlate the data with a template, weighting each frequency
# by `1/S_n(f)` so noisy frequencies count less,
#
# $$ \langle d \mid g\rangle \;=\; \mathrm{Re}\sum_f \frac{d_f^{*}\, g_f}{S_n(f)}. $$
#
# **Which templates?** A single matched filter against a fixed waveform
# is sharply peaked: it only responds to signals very close to that
# template (this is why real searches use banks of millions of
# templates). Instead we use the principled local choice from Lecture 2:
# linearise the waveform around a **fiducial** `θ₀` and project onto the
# two **score directions**, the derivatives `∂h/∂Mc` and `∂h/∂dL`. For
# Gaussian noise these two numbers are the locally-sufficient summary
# (the Fisher / Taylor picture). Intuitively they measure *"how much
# heavier than fiducial"* and *"how loud"*. The simulator provides the
# derivative templates; you build the filter.

# %% [markdown]
# ### ✏️ EXERCISE 1 — implement the matched-filter summary
#
# Implement `matched_filter`, then `my_summary`, using the two score
# templates and the PSD from the simulator. The simulator also provides
# a `whitening_matrix()` that rescales the two raw scores so that, under
# pure noise, the summary is independent `N(0, 1)`. Check that your
# summary reproduces `sim.summary(x)`.

# %%
g_mc, g_dl = sim.score_templates()      # derivative templates, freq-domain
psd = sim.psd(sim.freqs)                # noise PSD on the rfft grid
W = sim.whitening_matrix()              # 2x2, maps raw scores -> whitened summary


def matched_filter(x_td, template_fd, psd):
    # TODO — your code here.
    # 1. Xf = rfft of the time-domain data x_td
    # 2. return the real part of sum_f conj(Xf) * template_fd / psd
    raise NotImplementedError


def my_summary(x_td):
    # TODO — your code here.
    # 1. raw = [matched_filter(x, g_mc, psd), matched_filter(x, g_dl, psd)]
    # 2. return W @ raw
    raise NotImplementedError


# %%
# @title Reference solution { display-mode: "form" }
def matched_filter(x_td, template_fd, psd):  # noqa: F811
    Xf = np.fft.rfft(x_td)
    return float(np.real(np.sum(np.conj(Xf) * template_fd / psd)))


def my_summary(x_td):  # noqa: F811
    raw = np.array([matched_filter(x_td, g_mc, psd),
                    matched_filter(x_td, g_dl, psd)])
    return W @ raw


x_check = sim.simulate(np.array([30.0, 600.0]), rng=np.random.default_rng(3))
print("my_summary  :", my_summary(x_check).round(4))
print("sim.summary :", sim.summary(x_check).round(4))

# %% [markdown]
# **See the summary carry the parameters.** Below, each point is the
# summary of one noisy observation. Sweeping `Mc` slides the cloud along
# one direction; sweeping `dL` (loudness) slides it along the other.
# Under pure noise the cloud would sit at the origin with unit spread.

# %%
rng = np.random.default_rng(2)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# colour by Mc
for mc, c in zip([22, 30, 38, 44], plt.cm.viridis(np.linspace(0.1, 0.9, 4))):
    pts = np.array([sim.summary(sim.simulate(np.array([mc, dl]), rng=rng))
                    for dl in np.linspace(sim.dl_low, sim.dl_high, 6) for _ in range(8)])
    axes[0].scatter(pts[:, 0], pts[:, 1], s=8, color=c, alpha=0.6, label=fr"$M_c={mc}$")
# colour by dL
for dl, c in zip([400, 700, 1100], plt.cm.plasma(np.linspace(0.1, 0.8, 3))):
    pts = np.array([sim.summary(sim.simulate(np.array([mc, dl]), rng=rng))
                    for mc in np.linspace(sim.mc_low, sim.mc_high, 6) for _ in range(8)])
    axes[1].scatter(pts[:, 0], pts[:, 1], s=8, color=c, alpha=0.6, label=fr"$d_L={dl}$")
for ax, ttl in zip(axes, ["coloured by chirp mass", "coloured by distance"]):
    ax.axhline(0, color="0.8", lw=0.5); ax.axvline(0, color="0.8", lw=0.5)
    ax.set_xlabel(r"$s_1$"); ax.set_ylabel(r"$s_2$"); ax.legend(fontsize=8); ax.set_title(ttl)
fig.tight_layout(); plt.show()

# %% [markdown]
# ---
#
# ## 4 — A 2-D Gaussian head with a learned correlation
#
# Now the inference head. With two parameters the posterior is a 2-D
# distribution, and `Mc` and `dL` are **correlated** (a louder, lighter
# chirp can resemble a quieter, heavier one). A diagonal Gaussian could
# only produce axis-aligned ellipses; we give the head a **learned
# correlation coefficient** `ρ` so it can tilt.
#
# The head outputs five numbers per observation: the two means, the two
# log-variances (heteroscedastic, as in §1), and `ρ`. We squash `ρ`
# through `tanh` so it always lies in `(-1, 1)`.

# %% [markdown]
# ### ✏️ EXERCISE 2 — build the 2-D Gaussian head
#
# Fill in `forward` so it returns `(mu, log_var, rho)` with shapes
# `(batch, 2)`, `(batch, 2)`, `(batch, 1)`, and `ρ = tanh(...)`. The
# trunk and the three output layers are provided. The loss
# (`gaussian_nll_2d`, the correlated Gaussian NLL) is given below it;
# read it but you do not need to derive it.

# %%
class GaussianHead2D(nn.Module):
    def __init__(self, in_dim=2, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden, 2)
        self.head_logvar = nn.Linear(hidden, 2)
        self.head_rho = nn.Linear(hidden, 1)

    def forward(self, s):
        # TODO — your code here.
        # 1. h = self.trunk(s)
        # 2. mu = self.head_mu(h); log_var = self.head_logvar(h)
        # 3. rho = torch.tanh(self.head_rho(h))   # keep rho in (-1, 1)
        # 4. return mu, log_var, rho
        raise NotImplementedError


# %%
# @title Reference solution { display-mode: "form" }
class GaussianHead2D(nn.Module):  # noqa: F811
    def __init__(self, in_dim=2, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_mu = nn.Linear(hidden, 2)
        self.head_logvar = nn.Linear(hidden, 2)
        self.head_rho = nn.Linear(hidden, 1)

    def forward(self, s):
        h = self.trunk(s)
        mu = self.head_mu(h)
        log_var = self.head_logvar(h)
        rho = torch.tanh(self.head_rho(h))
        return mu, log_var, rho


def gaussian_nll_2d(theta, mu, log_var, rho):
    """Negative log-likelihood of a correlated 2-D Gaussian, batch mean.

    With standardised residuals z_i = (theta_i - mu_i) / sigma_i,

        -log N = 1/2 [ (z1^2 - 2 rho z1 z2 + z2^2) / (1 - rho^2)
                       + log sigma1^2 + log sigma2^2 + log(1 - rho^2) ].
    """
    rho = rho.squeeze(-1)
    z = (theta - mu) / torch.exp(0.5 * log_var)
    z1, z2 = z[:, 0], z[:, 1]
    one_minus = 1.0 - rho ** 2
    quad = (z1 ** 2 - 2.0 * rho * z1 * z2 + z2 ** 2) / one_minus
    logdet = log_var.sum(dim=-1) + torch.log(one_minus)
    return 0.5 * (quad + logdet).mean()


# %% [markdown]
# ### Training set and normalisation
#
# Simulate `(θ, summary)` pairs. Parameters and summaries live on very
# different scales, so we z-score both; the network works in normalised
# units and we map back when we plot. (`ρ` is unchanged by per-axis
# scaling, so it needs no correction.)

# %%
def make_dataset(n, rng):
    theta = sim.sample_prior(n, rng=rng)
    s = sim.summary(sim.simulate(theta, rng=rng, progress=True))
    return (torch.tensor(theta, dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32))


rng = np.random.default_rng(SEED)
theta_tr, s_tr = make_dataset(8000, rng)
theta_va, s_va = make_dataset(800, rng)

theta_mu, theta_sd = theta_tr.mean(0), theta_tr.std(0)
s_mu, s_sd = s_tr.mean(0), s_tr.std(0)
norm = lambda a, m, sd: (a - m) / sd
theta_tr_n, theta_va_n = norm(theta_tr, theta_mu, theta_sd), norm(theta_va, theta_mu, theta_sd)
s_tr_n, s_va_n = norm(s_tr, s_mu, s_sd), norm(s_va, s_mu, s_sd)

# %% [markdown]
# Train with the same five-step loop, now minimising `gaussian_nll_2d`.

# %%
torch.manual_seed(SEED)
model = GaussianHead2D()
opt = optim.Adam(model.parameters(), lr=1e-3)
n = s_tr_n.shape[0]
tr_curve, va_curve = [], []
bar = trange(150, desc="train 2-D head", leave=False)
for epoch in bar:
    perm = torch.randperm(n)
    for i in range(0, n, 256):
        idx = perm[i:i + 256]
        opt.zero_grad()
        loss = gaussian_nll_2d(theta_tr_n[idx], *model(s_tr_n[idx]))
        loss.backward(); opt.step()
    with torch.no_grad():
        tr_curve.append(gaussian_nll_2d(theta_tr_n, *model(s_tr_n)).item())
        va_curve.append(gaussian_nll_2d(theta_va_n, *model(s_va_n)).item())
    bar.set_postfix(train=f"{tr_curve[-1]:.3f}", val=f"{va_curve[-1]:.3f}")

plt.figure(figsize=(5, 3))
plt.plot(tr_curve, label="train"); plt.plot(va_curve, label="val")
plt.xlabel("epoch"); plt.ylabel("2-D Gaussian NLL"); plt.legend()
plt.title("training"); plt.tight_layout(); plt.show()

# %% [markdown]
# ---
#
# ## 5 — Validate against the exact reference posterior
#
# Because the summary is a *linear* function of the data, the posterior
# given the summary, `p(θ | s)`, is available analytically on a grid
# (`sim.true_posterior_summary`). This is the **best possible** posterior
# given the same two summaries, so any gap is purely our Gaussian
# approximation, not lost information.
#
# We evaluate the trained head and the reference at a loud observation
# and overlay them, plus the 1-D marginals.

# %%
def npe_gaussian_params(model, s_obs):
    s_n = norm(torch.tensor(s_obs, dtype=torch.float32), s_mu, s_sd)
    with torch.no_grad():
        mu, lv, rho = model(s_n.unsqueeze(0))
    mu = (mu.squeeze(0) * theta_sd + theta_mu).numpy()
    sig = (torch.exp(0.5 * lv.squeeze(0)) * theta_sd).numpy()
    return mu, sig, float(rho.squeeze())


def gaussian2d_grid(mc_g, dl_g, mu, sig, rho):
    MC, DL = np.meshgrid(mc_g, dl_g, indexing="ij")
    z1 = (MC - mu[0]) / sig[0]; z2 = (DL - mu[1]) / sig[1]
    q = (z1 ** 2 - 2 * rho * z1 * z2 + z2 ** 2) / (1 - rho ** 2)
    return np.exp(-0.5 * q)


def show_posterior(theta_t, seed, title):
    x = sim.simulate(np.array(theta_t), rng=np.random.default_rng(seed))
    s_obs = sim.summary(x)
    mc_g, dl_g, p_ref = sim.true_posterior_summary(s_obs, n_mc=120, n_dl=120)
    mu, sig, rho = npe_gaussian_params(model, s_obs)
    p_npe = gaussian2d_grid(mc_g, dl_g, mu, sig, rho)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1)
    # contour the 68% and 95% credible regions, not arbitrary iso-density lines
    ax.contour(*np.meshgrid(mc_g, dl_g, indexing="ij"), p_ref,
               levels=credible_levels(p_ref), colors="k", linewidths=0.8)
    ax.contour(*np.meshgrid(mc_g, dl_g, indexing="ij"), p_npe,
               levels=credible_levels(p_npe), colors="C0", linewidths=1.2)
    ax.plot(*theta_t, "C3*", ms=13)
    ax.set_xlabel(r"$M_c\ [M_\odot]$"); ax.set_ylabel(r"$d_L$ [Mpc]")
    ax.set_title(f"{title}\nblack = exact, blue = NPE  (68% / 95%, rho={rho:+.2f})")
    # marginals
    axm = fig.add_subplot(2, 2, 2)
    pm = p_ref.sum(1); pm = pm / (pm.sum() * (mc_g[1] - mc_g[0]))
    axm.plot(mc_g, pm, "k-", lw=1.2)
    axm.plot(mc_g, np.exp(-0.5 * ((mc_g - mu[0]) / sig[0]) ** 2) / (sig[0] * np.sqrt(2 * np.pi)),
             "C0--", lw=1.2)
    axm.set_xlabel(r"$M_c$"); axm.set_title("marginal $M_c$", fontsize=9); axm.set_yticks([])
    axd = fig.add_subplot(2, 2, 4)
    pd = p_ref.sum(0); pd = pd / (pd.sum() * (dl_g[1] - dl_g[0]))
    axd.plot(dl_g, pd, "k-", lw=1.2)
    axd.plot(dl_g, np.exp(-0.5 * ((dl_g - mu[1]) / sig[1]) ** 2) / (sig[1] * np.sqrt(2 * np.pi)),
             "C0--", lw=1.2)
    axd.set_xlabel(r"$d_L$"); axd.set_title("marginal $d_L$", fontsize=9); axd.set_yticks([])
    fig.tight_layout(); plt.show()


show_posterior((38.0, 500.0), seed=7, title="loud signal (SNR ~ 23)")

# %% [markdown]
# The blue NPE contours sit on the black reference, **tilt and all**: the
# learned correlation `ρ` lets the Gaussian follow the diagonal
# degeneracy, and the 1-D marginals line up. The single trained network
# does this for *any* observation in one forward pass (amortised): no
# per-event refitting.
#
# ### ✏️ EXERCISE 3 — find where the Gaussian starts to strain
#
# Re-run `show_posterior` for fainter signals (push `dL` toward the far
# edge of the prior, e.g. `(40, 1000)` or `(25, 1050)`). At low SNR the
# exact posterior stops being a clean tilted ellipse and starts to
# **curve** (a banana). The correlated Gaussian captures the tilt but not
# the curvature. Where does the mismatch first become visible?

# %%
# TODO — your code here, e.g.:
# show_posterior((40.0, 1000.0), seed=11, title="faint signal")


# %%
# @title Reference solution { display-mode: "form" }
show_posterior((40.0, 1000.0), seed=11, title="faint signal (SNR ~ 12)")

# %% [markdown]
# ---
#
# ## 6 — Preview: a normalising flow (for fun)
#
# A **normalising flow** is a far more flexible `q_φ(θ|x)`: instead of a
# Gaussian shape it learns to bend a simple distribution into whatever
# the posterior actually looks like, curvature and all. It is the
# centrepiece of the next session. As a teaser, here is one trained on
# the *same* `(θ, summary)` pairs using the `sbi` library, in a few
# lines. (Optional: the install is heavy and may ask for a runtime
# restart. Skip if you are short on time.)

# %%
# @title Optional flow preview (installs sbi) { display-mode: "form" }
try:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sbi"], check=True)
    from sbi.inference import SNPE
    from sbi.utils import BoxUniform

    prior = BoxUniform(low=torch.tensor([sim.mc_low, sim.dl_low]),
                       high=torch.tensor([sim.mc_high, sim.dl_high]))
    inference = SNPE(prior=prior)
    inference.append_simulations(theta_tr, s_tr)     # (theta, summary) pairs
    inference.train()
    flow_posterior = inference.build_posterior()

    theta_t = (40.0, 1000.0)
    x = sim.simulate(np.array(theta_t), rng=np.random.default_rng(11))
    s_obs = sim.summary(x)
    samples = flow_posterior.sample((3000,), x=torch.tensor(s_obs, dtype=torch.float32)).numpy()

    mc_g, dl_g, p_ref = sim.true_posterior_summary(s_obs, n_mc=120, n_dl=120)
    plt.figure(figsize=(5.5, 4.5))
    plt.contour(*np.meshgrid(mc_g, dl_g, indexing="ij"), p_ref,
                levels=credible_levels(p_ref), colors="k", linewidths=0.8)
    plt.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.15, color="C2")
    plt.plot(*theta_t, "C3*", ms=13)
    plt.xlabel(r"$M_c\ [M_\odot]$"); plt.ylabel(r"$d_L$ [Mpc]")
    plt.title("flow samples (green) vs exact posterior (black)")
    plt.xlim(sim.mc_low, sim.mc_high); plt.ylim(sim.dl_low, sim.dl_high)
    plt.tight_layout(); plt.show()
except Exception as e:
    print("Flow preview skipped:", type(e).__name__, e)
    print("This is optional — the session does not depend on it.")

# %% [markdown]
# The flow samples fill the curved reference posterior, including the
# bend a single Gaussian cannot follow. That flexibility, plus learned
# summary networks and calibration diagnostics, is what the next session
# is about.
#
# ## Where this lands you
#
# - You fixed the Session-1A limitation: a width that depends on `x`.
# - You turned an 8192-sample time series into a 2-number summary by
#   hand, via matched filtering against the score directions.
# - You built a 2-D Gaussian head with a learned correlation and used it
#   as an amortised posterior estimator on a realistic GW problem,
#   validated against the exact reference posterior.
# - You saw the one shape it still cannot make (curvature), and the flow
#   that can.
