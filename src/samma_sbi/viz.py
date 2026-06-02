"""Visualisation helpers.

Pedagogical figures that students twist knobs on, kept out of the
notebook so the notebook itself stays narrative-heavy and code-light.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc


def plot_cannon_throw(
    theta: float,
    v0: float = 10.0,
    sigma_v: float = 0.0,
    sigma_x: float = 0.3,
    n_samples: int = 5,
    g: float = 9.81,
    seed: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw the cannon, parabolic trajectories, and noisy landing points.

    A pedagogical illustration of the ball-throw simulator. Each call
    samples ``n_samples`` trajectories with launch speed jittered by
    ``sigma_v`` and a Gaussian measurement error ``sigma_x`` on the
    landing position. The noise-free trajectory and landing position are
    shown for reference.

    Parameters
    ----------
    theta     : launch angle in radians.
    v0        : nominal launch speed (m/s).
    sigma_v   : standard deviation of v0 jitter (m/s). 0 means fixed v0.
    sigma_x   : standard deviation of landing-position measurement noise.
    n_samples : number of trajectories to draw.
    g         : gravitational acceleration.
    seed      : RNG seed; pass an integer to keep the picture frozen.
    ax        : matplotlib axes to draw into. New figure if None.

    Returns
    -------
    The axes object the figure was drawn into.
    """
    rng = np.random.default_rng(seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))

    # Reference: noise-free range and trajectory at the nominal v0.
    range_nom = (v0 ** 2 / g) * np.sin(2 * theta)
    t_land_nom = 2 * v0 * np.sin(theta) / g
    t_nom = np.linspace(0, t_land_nom, 200)
    x_nom = v0 * np.cos(theta) * t_nom
    y_nom = v0 * np.sin(theta) * t_nom - 0.5 * g * t_nom ** 2

    # Sample trajectories with v jitter and landing-noise.
    v_samples = np.maximum(rng.normal(v0, sigma_v, size=n_samples), 0.1)
    noise_samples = rng.normal(0.0, sigma_x, size=n_samples)
    landings = []
    for v_i, n_i in zip(v_samples, noise_samples):
        t_land = 2 * v_i * np.sin(theta) / g
        t = np.linspace(0, t_land, 200)
        x_traj = v_i * np.cos(theta) * t
        y_traj = v_i * np.sin(theta) * t - 0.5 * g * t ** 2
        ax.plot(x_traj, y_traj, lw=1.0, alpha=0.45, color="C0")
        landing = x_traj[-1] + n_i
        landings.append(landing)
        ax.plot(landing, 0, "o", color="red", markersize=9,
                markeredgecolor="darkred", zorder=5, alpha=0.85)

    landings = np.asarray(landings)

    # Reference trajectory on top.
    ax.plot(x_nom, y_nom, lw=2.0, color="C0", alpha=0.9, zorder=4,
            label="noise-free trajectory")

    # Ground line.
    x_left, x_right = -1.5, max(range_nom, landings.max() if landings.size else 0) * 1.15 + 1
    ax.plot([x_left, x_right], [0, 0], color="black", lw=1.0, zorder=1)

    # Vertical dashed line at the noise-free landing position.
    y_max_nom = (v0 * np.sin(theta)) ** 2 / (2 * g)
    ax.plot([range_nom, range_nom], [0, 0.15 * y_max_nom + 0.5],
            color="C0", ls=":", lw=1.0)
    ax.annotate(r"$r(\theta)$",
                xy=(range_nom, 0.15 * y_max_nom + 0.6),
                ha="center", va="bottom", fontsize=10, color="C0")

    # Cannon: a thick wedge along the launch direction.
    cannon_len = 1.2
    cannon_x = cannon_len * np.cos(theta)
    cannon_y = cannon_len * np.sin(theta)
    ax.plot([0, cannon_x], [0, cannon_y],
            lw=10, color="dimgray", solid_capstyle="round", zorder=2)
    # Cannon base.
    ax.add_patch(plt.Rectangle((-0.55, -0.4), 1.1, 0.4,
                               color="dimgray", zorder=3))
    # Wheel.
    ax.add_patch(plt.Circle((0.0, -0.3), 0.35,
                            facecolor="saddlebrown",
                            edgecolor="black", lw=1.0, zorder=4))

    # theta arc + label.
    arc_r = 0.65
    arc = Arc((0, 0), 2 * arc_r, 2 * arc_r, angle=0,
              theta1=0, theta2=np.degrees(theta), color="black", lw=1.5)
    ax.add_patch(arc)
    ax.annotate(r"$\theta$",
                xy=((arc_r + 0.18) * np.cos(theta / 2),
                    (arc_r + 0.18) * np.sin(theta / 2)),
                fontsize=14, ha="center", va="center")

    # Annotate measurement style.
    ax.plot([], [], "o", color="red", markersize=9, markeredgecolor="darkred",
            label=f"measured landings (×{n_samples})")
    ax.plot([], [], lw=1.0, color="C0", alpha=0.5,
            label=f"sampled trajectories ($\\sigma_v$={sigma_v:.2f})")

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(-0.9, max(y_max_nom * 1.4, 2.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        rf"$\theta = {np.degrees(theta):.1f}°$,  "
        rf"$v_0 = {v0:.2f} \pm {sigma_v:.2f}$ m/s,  "
        rf"$\sigma_x = {sigma_x:.2f}$ m"
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    return ax


__all__ = ["plot_cannon_throw"]
