# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cweniger/teaching-2606-ICTP-SAIFR/blob/main/notebooks/lec1b_abc_demos.ipynb)
#
# # Lec 1b — Classic ABC: three demos on the ball-throw
#
# Reference notebook for the three live demos in Lecture 1b. Each demo
# uses the same one-parameter ball-throw simulator and the same observed
# landing position `x_obs`.
#
# **Pedagogical staircase:**
#
# 1. **Rejection ABC.** Keep the top-`M` of `N` simulations by distance
#    to `x_obs`. Sliders on `M` and `N` show the sample-efficiency cliff.
# 2. **+ Summary statistic.** Throw `n_balls` per `theta` and use their
#    mean as the summary. Same `M`, `N`. The posterior tightens
#    dramatically at fixed `N`.
# 3. **KDE in (theta, x)-space.** No accept/reject. Build a 2D KDE on the
#    full simulation cloud and slice it at `x_obs`. Solves the
#    epsilon-zero limit cleanly in 1D — and visibly fails when `x` gains
#    dummy dimensions.
#
# The notebook doubles as the algorithmic reference for the JS port that
# will live in the reveal.js Lec 1b slides.

# %%
# Always reinstall to pick up the latest from main — pip would otherwise
# short-circuit on an already-installed version, even when git HEAD has
# changed. --no-deps skips numpy/scipy/matplotlib (Colab ships them).
# !pip install -q --upgrade --force-reinstall --no-deps git+https://github.com/cweniger/teaching-2606-ICTP-SAIFR.git  # noqa: E501

# %%
import numpy as np
import matplotlib.pyplot as plt

from samma_sbi.simulators import BallThrow
from samma_sbi.abc import (
    rejection_abc,
    joint_kde_posterior,
    joint_kde_posterior_highd,
)
from samma_sbi.viz import plot_cannon_throw

# A single RNG seed for reproducibility across the whole notebook.
SEED = 0

# %% [markdown]
# Figures are saved to `docs/figures/lec1b/` as both PDF (vector, for
# LaTeX inclusion in lecture notes) and PNG (for slides / web). Saving is
# skipped silently in Colab.

# %%
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if not IN_COLAB:
    _here = Path.cwd().resolve()
    REPO_ROOT = next(
        (p for p in [_here, *_here.parents] if (p / "pyproject.toml").exists()),
        _here,
    )
    FIG_DIR = REPO_ROOT / "docs" / "figures" / "lec1b"
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(fig, name: str) -> None:
    """Save *fig* as ``docs/figures/lec1b/{name}.pdf`` and ``.png``.

    No-op in Colab (the directory wouldn't be persisted anyway).
    """
    if IN_COLAB:
        return
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## The simulator
#
# One parameter `theta` (launch angle, radians). One observation `x`
# (landing position in metres). The expected range is
# $r(\theta) = (v_0^2 / g)\sin(2\theta)$, plus Gaussian noise.
#
# **Important geometric fact.** Because $\sin(2\theta)$ is symmetric
# around $\theta = \pi/4$, the mapping $\theta \mapsto r(\theta)$ is
# two-to-one on $(0, \pi/2)$. The posterior is therefore generically
# bimodal — the cleanest possible motivation for going beyond
# Gaussian-head NPE later in the school.

# %%
sim = BallThrow(v0=10.0, g=9.81, sigma=0.3)

# Pick a ground-truth theta and the corresponding x_obs.
rng = np.random.default_rng(SEED)
theta_true = 0.6  # radians (~34 degrees)
x_obs = float(sim.simulate(np.array([theta_true]), rng=rng).item())
print(f"theta_true = {theta_true:.3f} rad ({np.degrees(theta_true):.1f} deg)")
print(f"x_obs      = {x_obs:.3f} m")
print(f"r(theta_true) = {sim.range_mean(theta_true):.3f} m  (noise-free)")

# %% [markdown]
# ### What the simulator actually does — pedagogical picture
#
# A cannon at the origin, launch angle $\theta$, launch speed $v_0$, and
# a Gaussian measurement error $\sigma_x$ on the landing position. The
# panels below show single-throw, multi-throw with measurement noise,
# and multi-throw with both $v_0$ jitter and measurement noise.

# %%
fig, axs = plt.subplots(3, 1, figsize=(10, 11))
plot_cannon_throw(theta=theta_true, v0=10.0, sigma_v=0.0, sigma_x=0.0,
                  n_samples=1, seed=SEED, ax=axs[0])
axs[0].set_title(rf"Single noise-free throw, $\theta = {np.degrees(theta_true):.0f}°$")
plot_cannon_throw(theta=theta_true, v0=10.0, sigma_v=0.0, sigma_x=0.3,
                  n_samples=10, seed=SEED, ax=axs[1])
axs[1].set_title(rf"10 throws, measurement noise only ($\sigma_x = 0.30$ m)")
plot_cannon_throw(theta=theta_true, v0=10.0, sigma_v=0.3, sigma_x=0.3,
                  n_samples=10, seed=SEED, ax=axs[2])
axs[2].set_title(rf"10 throws, with $v_0$ jitter ($\sigma_v = 0.30$ m/s)")
fig.tight_layout()
savefig(fig, "cannon_overview")
plt.show()

# %% [markdown]
# Quick visual of the simulator — the range function with a noise band,
# and the reference posterior (analytic, on a fine grid) overlaid for a
# fixed `x_obs`.

# %%
theta_grid = np.linspace(sim.prior_low, sim.prior_high, 1001)
r_grid = sim.range_mean(theta_grid)
_, true_post = sim.true_posterior(x_obs, theta_grid=theta_grid)

fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(11, 4))

ax_r.plot(theta_grid, r_grid, lw=2)
ax_r.fill_between(
    theta_grid, r_grid - sim.sigma, r_grid + sim.sigma, alpha=0.2,
    label=r"$\pm\sigma$",
)
ax_r.axhline(x_obs, color="k", ls="--", lw=1, label=r"$x_\mathrm{obs}$")
ax_r.set_xlabel(r"$\theta$ (rad)")
ax_r.set_ylabel(r"landing position $x$ (m)")
ax_r.set_title("Forward model")
ax_r.legend()

ax_p.plot(theta_grid, true_post, lw=2, color="C3")
ax_p.axvline(theta_true, color="k", ls="--", lw=1, label=r"$\theta_\mathrm{true}$")
ax_p.set_xlabel(r"$\theta$ (rad)")
ax_p.set_ylabel(r"$p(\theta \mid x_\mathrm{obs})$")
ax_p.set_title(f"Reference posterior at $x_\\mathrm{{obs}} = {x_obs:.2f}$")
ax_p.legend()

fig.tight_layout()
savefig(fig, "simulator_overview")
plt.show()

# %% [markdown]
# Two peaks, as expected. Every demo below should recover something close
# to this red curve.

# %% [markdown]
# ## Demo 1 — Rejection ABC
#
# Draw `N` samples from the prior, simulate one ball per sample, keep the
# `M` with the smallest distance to `x_obs`. The histogram of accepted
# `theta` values estimates the posterior.
#
# Knobs to twist:
#
# - **Small `M`**: histogram is noisy.
# - **Large `M` (approaching `N`)**: accepts everything, posterior leaks
#   toward the prior.
# - **Small `N`**: very few sims land near `x_obs`, so even the top-`M`
#   are far from `x_obs`. Waste.

# %%
def demo1(N: int, M: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    theta = sim.sample_prior(N, rng=rng)
    x = sim.simulate(theta, n_balls=1, rng=rng)
    accepted = rejection_abc(theta, x, x_obs, M=M)
    return theta, x, accepted


def plot_demo1(N: int, M: int, ax=None, seed: int = SEED):
    theta, x, accepted = demo1(N=N, M=M, seed=seed)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        accepted, bins=40, range=(sim.prior_low, sim.prior_high),
        density=True, alpha=0.6, label=f"ABC (N={N}, M={M})",
    )
    ax.plot(theta_grid, true_post, lw=2, color="C3", label="reference")
    ax.axvline(theta_true, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel("density")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)
    return ax


fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
plot_demo1(N=200, M=20, ax=axs[0, 0])
axs[0, 0].set_title("N=200, M=20 (noisy)")
plot_demo1(N=2000, M=200, ax=axs[0, 1])
axs[0, 1].set_title("N=2000, M=200 (cleaner)")
plot_demo1(N=2000, M=1800, ax=axs[1, 0])
axs[1, 0].set_title("N=2000, M=1800 (prior leakage)")
plot_demo1(N=20000, M=200, ax=axs[1, 1])
axs[1, 1].set_title("N=20000, M=200 (sharp, expensive)")
fig.suptitle("Rejection ABC — the M/N tradeoff", y=1.02)
fig.tight_layout()
savefig(fig, "demo1_rejection_abc")
plt.show()

# %% [markdown]
# **Key message.** The shape converges to the bimodal reference posterior
# only when `N` is large enough that the top-`M` are tightly clustered
# around `x_obs`. The simulator pays for that with brute-force sim count.

# %% [markdown]
# ## Demo 2 — Add a summary statistic
#
# Same algorithm, but each `theta` now generates `n_balls` simulated
# landings and the summary is their mean. Variance of the summary scales
# as $\sigma^2 / n_\mathrm{balls}$, so accepted samples cluster much
# more tightly at the same `M`, `N`.

# %%
def demo2(N: int, M: int, n_balls: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    theta = sim.sample_prior(N, rng=rng)
    x = sim.simulate_summary(theta, n_balls=n_balls, rng=rng)
    accepted = rejection_abc(theta, x, x_obs, M=M)
    return theta, x, accepted


def plot_demo2(N: int, M: int, n_balls: int, ax=None, seed: int = SEED):
    theta, x, accepted = demo2(N=N, M=M, n_balls=n_balls, seed=seed)
    # Reference posterior with the corresponding summary variance.
    _, post_ref = sim.true_posterior(
        x_obs, theta_grid=theta_grid, n_balls=n_balls
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        accepted, bins=40, range=(sim.prior_low, sim.prior_high),
        density=True, alpha=0.6,
        label=f"ABC (n_balls={n_balls})",
    )
    ax.plot(theta_grid, post_ref, lw=2, color="C3", label="reference")
    ax.axvline(theta_true, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel("density")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)
    return ax


fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True, sharex=True)
plot_demo2(N=2000, M=200, n_balls=1,   ax=axs[0])
axs[0].set_title("n_balls = 1 (= Demo 1)")
plot_demo2(N=2000, M=200, n_balls=10,  ax=axs[1])
axs[1].set_title("n_balls = 10")
plot_demo2(N=2000, M=200, n_balls=100, ax=axs[2])
axs[2].set_title("n_balls = 100")
fig.suptitle("Adding a summary statistic — same N, M, much sharper posterior",
             y=1.05)
fig.tight_layout()
savefig(fig, "demo2_summary")
plt.show()

# %% [markdown]
# **Key message.** A well-chosen summary buys orders of magnitude in
# sample efficiency. Open question for later in the school: who chooses
# the summary when `x` is high-dimensional?

# %% [markdown]
# ## Demo 3 — KDE in (theta, x)-space
#
# No accept/reject. Build a 2D kernel-density estimate on the full
# simulation cloud and slice it at `x_obs`. Two virtues:
#
# - Uses *all* `N` simulations (no rejected sims wasted).
# - Recovers the posterior cleanly without an epsilon to tune.
#
# This is what the $\epsilon \to 0$ limit of ABC looks like done right —
# and the conceptual seed of every neural-SBI method that follows.

# %%
def demo3(N: int, bandwidth="scott", seed: int = SEED):
    rng = np.random.default_rng(seed)
    theta = sim.sample_prior(N, rng=rng)
    x = sim.simulate(theta, n_balls=1, rng=rng)
    post = joint_kde_posterior(theta, x, x_obs, theta_grid, bandwidth=bandwidth)
    return theta, x, post


def plot_demo3(N: int, bandwidth="scott", seed: int = SEED):
    theta, x, post = demo3(N=N, bandwidth=bandwidth, seed=seed)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: scatter of (theta, x) with x_obs slice highlighted.
    axs[0].scatter(theta, x, s=4, alpha=0.3, color="C0", label="sims")
    axs[0].axhline(x_obs, color="k", ls="--", lw=1, label=r"$x_\mathrm{obs}$")
    axs[0].plot(theta_grid, sim.range_mean(theta_grid),
                color="C1", lw=1, label=r"$r(\theta)$")
    axs[0].set_xlabel(r"$\theta$")
    axs[0].set_ylabel(r"$x$")
    axs[0].set_title(f"Joint sims (N={N})")
    axs[0].legend(loc="upper right", fontsize=9)

    # Right: KDE conditional slice vs reference posterior.
    axs[1].plot(theta_grid, post, color="C2", lw=2, label="KDE slice")
    axs[1].plot(theta_grid, true_post, color="C3", lw=2, ls="--",
                label="reference")
    axs[1].axvline(theta_true, color="k", ls="--", lw=1)
    axs[1].set_xlabel(r"$\theta$")
    axs[1].set_ylabel(r"$p(\theta \mid x_\mathrm{obs})$")
    axs[1].set_title(f"Conditional slice (bandwidth={bandwidth!r})")
    axs[1].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


for N in (500, 2000):
    fig = plot_demo3(N=N, bandwidth="scott")
    savefig(fig, f"demo3_kde_n{N}")
    plt.show()

# %% [markdown]
# **Key message.** At the same `N` as Demo 1, the KDE slice is
# dramatically cleaner — the bimodal structure is captured rather than
# histogrammed-against. No epsilon to tune.

# %% [markdown]
# ## Demo 3b — Failure mode in higher dimensions
#
# The KDE trick works gracefully in 1D `x`. Pad `x_obs` with dummy
# Gaussian noise dimensions (independent of `theta`, so they contain
# *no information*) and the bandwidth choice collapses the estimate.

# %%
def demo3_highd(N: int, n_dummy: int, bandwidth="scott", seed: int = SEED):
    rng = np.random.default_rng(seed)
    theta = sim.sample_prior(N, rng=rng)
    x_true = sim.simulate(theta, n_balls=1, rng=rng)
    if n_dummy > 0:
        dummies = rng.standard_normal((N, n_dummy))
        x = np.column_stack([x_true, dummies])
        x_obs_padded = np.concatenate([[x_obs], np.zeros(n_dummy)])
    else:
        x = x_true[:, None]
        x_obs_padded = np.array([x_obs])
    post = joint_kde_posterior_highd(
        theta, x, x_obs_padded, theta_grid, bandwidth=bandwidth
    )
    return post


fig, ax = plt.subplots(figsize=(7, 4.5))
for n_dummy in [0, 1, 5, 20]:
    post = demo3_highd(N=2000, n_dummy=n_dummy, bandwidth="scott")
    ax.plot(theta_grid, post, lw=2, label=f"dim(x) = {1 + n_dummy}")
ax.plot(theta_grid, true_post, lw=2, color="k", ls="--", label="reference")
ax.axvline(theta_true, color="grey", ls=":", lw=1)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$p(\theta \mid x_\mathrm{obs})$")
ax.set_title("KDE posterior as we pad x_obs with uninformative dimensions")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
savefig(fig, "demo3b_curse_of_dimensionality")
plt.show()

# %% [markdown]
# **Key message — the school's syllabus in one figure.** Even though the
# added dimensions carry *zero information*, the KDE estimate degrades
# rapidly. Two distinct failures need solving:
#
# 1. **Summary identification.** When `x` is high-dimensional and
#    structured, *which* function of `x` should we condition on?
#    → Lecture 4: neural summary networks.
# 2. **Density estimation in high D.** Even after a good summary, fitting
#    a density in many dimensions is hard.
#    → Lecture 3: normalising flows / NPE.
#
# Everything else in the school is in service of these two problems.
#
# ---
#
# Live, slider-driven exploration of these demos lives in the reveal.js
# slides (JS implementation). This notebook is the canonical source for
# the algorithms and the figure factory for the frozen lecture-notes
# PDFs.
