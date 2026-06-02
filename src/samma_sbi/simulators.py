"""Toy + APP-flavoured simulators used across the school.

Each simulator exposes a small uniform API:

    sim = SomeSim(...)
    theta = sim.sample_prior(n)            # draw parameters from the prior
    x = sim.simulate(theta, n_balls=...)   # forward-model
    p_grid = sim.true_posterior(x_obs)     # reference posterior on a grid

Vectorise wherever possible — ABC needs N >> 1 simulator calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid


@dataclass
class BallThrow:
    """One-parameter ball-throw simulator.

    A projectile is launched at angle ``theta`` with fixed speed ``v0`` from
    ground level on a flat planet of gravity ``g``. The range is

        r(theta) = (v0 ** 2 / g) * sin(2 * theta)

    Observations are landing positions plus Gaussian noise of standard
    deviation ``sigma``. The prior on ``theta`` is uniform over
    ``[prior_low, prior_high]``.

    The mapping theta -> r is two-to-one on (0, pi/2): both ``pi/4 - delta``
    and ``pi/4 + delta`` land at the same expected position. The posterior
    is therefore bimodal in general — a feature, not a bug. It sets up:

    * the visual interest of the ABC demos (Lec 1b),
    * the failure of Gaussian-head NPE on bimodal posteriors (Lec 2 -> Lec 3),
    * the natural motivation for normalising flows (Lec 3).
    """

    v0: float = 10.0        # m/s
    g: float = 9.81         # m/s^2
    sigma: float = 0.3      # noise on landing position, in metres
    prior_low: float = 0.05            # radians; avoid exactly 0
    prior_high: float = np.pi / 2 - 0.05  # avoid exactly pi/2

    # ------------------------------------------------------------------
    # Core forward model
    # ------------------------------------------------------------------
    def range_mean(self, theta: np.ndarray | float) -> np.ndarray | float:
        """Noise-free expected landing position r(theta)."""
        return (self.v0 ** 2 / self.g) * np.sin(2.0 * np.asarray(theta))

    def simulate(
        self,
        theta: np.ndarray | float,
        n_balls: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Forward-simulate ``n_balls`` landing positions per theta.

        Returns shape ``(*theta.shape, n_balls)`` if n_balls > 1, otherwise
        ``theta.shape``. With ``n_balls == 1`` the trailing axis is squeezed.
        """
        rng = np.random.default_rng() if rng is None else rng
        theta = np.asarray(theta)
        mu = self.range_mean(theta)
        if n_balls == 1:
            return mu + self.sigma * rng.standard_normal(mu.shape)
        # Broadcast: mu has shape S, draw S x n_balls noise
        noise = self.sigma * rng.standard_normal(mu.shape + (n_balls,))
        return mu[..., None] + noise

    def simulate_summary(
        self,
        theta: np.ndarray | float,
        n_balls: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Forward-simulate and return the per-theta mean of n_balls landings."""
        x = self.simulate(theta, n_balls=n_balls, rng=rng)
        if n_balls == 1:
            return x
        return x.mean(axis=-1)

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------
    def sample_prior(
        self, n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return rng.uniform(self.prior_low, self.prior_high, size=n)

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        """Log uniform prior; -inf outside the support."""
        theta = np.asarray(theta)
        inside = (theta >= self.prior_low) & (theta <= self.prior_high)
        out = np.full(theta.shape, -np.inf)
        out[inside] = -np.log(self.prior_high - self.prior_low)
        return out

    # ------------------------------------------------------------------
    # Reference posterior (analytic up to a normalising integral)
    # ------------------------------------------------------------------
    def true_posterior(
        self,
        x_obs: float | np.ndarray,
        n_balls: int = 1,
        theta_grid: np.ndarray | None = None,
        n_grid: int = 1001,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reference posterior p(theta | x_obs) on a fine theta grid.

        With Gaussian noise of variance sigma^2 / n_balls on the summary and
        a uniform prior, the posterior is proportional to the likelihood:

            p(theta | x_obs) ~ N(x_obs; r(theta), sigma^2 / n_balls) * 1[prior]

        Normalises on the supplied grid by trapezoidal integration.

        Returns
        -------
        theta_grid : array of theta values
        posterior  : normalised posterior density on the grid
        """
        if theta_grid is None:
            theta_grid = np.linspace(self.prior_low, self.prior_high, n_grid)
        mu = self.range_mean(theta_grid)
        var = self.sigma ** 2 / n_balls
        log_like = -0.5 * ((x_obs - mu) ** 2 / var + np.log(2 * np.pi * var))
        # Uniform prior -> log_prior is a constant on the grid
        log_post = log_like
        log_post -= log_post.max()  # numerical safety
        post = np.exp(log_post)
        post /= trapezoid(post, theta_grid)
        return theta_grid, post


__all__ = ["BallThrow"]
