"""Classic ABC algorithms used in Lec 1b.

Three demos:

* ``rejection_abc``        — keep top-M of N by distance to x_obs.
* ``summary_abc``          — same, but applied to a summary statistic.
* ``joint_kde_posterior``  — KDE on the joint (theta, x) cloud, then
                             condition by slicing at x_obs.

Implementations are intentionally small and direct — these are reference
implementations for the lecture, not high-performance code.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde


def rejection_abc(
    theta: np.ndarray,
    x: np.ndarray,
    x_obs: float,
    M: int,
) -> np.ndarray:
    """Keep the M sims whose simulated x is closest to x_obs.

    Parameters
    ----------
    theta : (N,) array of parameter samples drawn from the prior.
    x     : (N,) array of corresponding simulator outputs (or summaries).
    x_obs : observed value.
    M     : how many to keep.

    Returns
    -------
    (M,) array of accepted theta samples.

    Notes
    -----
    Using top-M rather than a distance threshold epsilon means the demo
    always has something to plot — the failure mode (large M -> prior
    leakage, small M -> noisy posterior) is what students should observe.
    """
    if M > theta.size:
        raise ValueError(f"M={M} exceeds N={theta.size}.")
    dists = np.abs(x - x_obs)
    keep_idx = np.argpartition(dists, M)[:M]
    return theta[keep_idx]


def joint_kde_posterior(
    theta: np.ndarray,
    x: np.ndarray,
    x_obs: float,
    theta_grid: np.ndarray,
    bandwidth: str | float = "scott",
) -> np.ndarray:
    """Posterior estimate via a 2D KDE on (theta, x), sliced at x_obs.

    This uses *all* N simulations (no accept/reject), and so does not have
    an epsilon problem in the small-bandwidth limit. The price is the
    curse of dimensionality: when x grows beyond 1-2 dimensions, the
    bandwidth choice destroys the estimate.

    Parameters
    ----------
    theta, x  : (N,) arrays.
    x_obs     : observed value of x.
    theta_grid: (G,) grid on which to evaluate the conditional density.
    bandwidth : passed to scipy.stats.gaussian_kde.

    Returns
    -------
    (G,) array of (renormalised) p(theta | x_obs) values.
    """
    points = np.vstack([theta, x])
    kde = gaussian_kde(points, bw_method=bandwidth)
    eval_points = np.vstack([theta_grid, np.full_like(theta_grid, x_obs)])
    joint = kde(eval_points)
    # Normalise the conditional slice on theta_grid
    joint = np.maximum(joint, 0.0)
    norm = trapezoid(joint, theta_grid)
    if norm <= 0:
        return joint
    return joint / norm


def joint_kde_posterior_highd(
    theta: np.ndarray,
    x: np.ndarray,
    x_obs: np.ndarray,
    theta_grid: np.ndarray,
    bandwidth: str | float = "scott",
) -> np.ndarray:
    """Same as ``joint_kde_posterior`` but x is multi-dimensional.

    Used in Lec 1b's failure-mode demo: pad x_obs with dummy dimensions
    and watch the bandwidth choice destroy the estimate.

    Parameters
    ----------
    theta : (N,) array.
    x     : (N, D) array of simulator outputs.
    x_obs : (D,) observed vector.
    """
    x = np.atleast_2d(x)
    x_obs = np.atleast_1d(x_obs)
    if x.shape[1] != x_obs.size:
        raise ValueError(
            f"x has D={x.shape[1]}, x_obs has D={x_obs.size}; mismatch."
        )
    points = np.vstack([theta, x.T])  # shape (1 + D, N)
    kde = gaussian_kde(points, bw_method=bandwidth)
    G = theta_grid.size
    eval_points = np.vstack([theta_grid, np.tile(x_obs[:, None], (1, G))])
    joint = kde(eval_points)
    joint = np.maximum(joint, 0.0)
    norm = trapezoid(joint, theta_grid)
    if norm <= 0:
        return joint
    return joint / norm


__all__ = [
    "rejection_abc",
    "joint_kde_posterior",
    "joint_kde_posterior_highd",
]
