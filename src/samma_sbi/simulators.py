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


# =====================================================================
# Gravitational-wave chirp simulator (Session 1, Block 3 — Option A)
# =====================================================================
#
# Equal-mass inspiral-merger-ringdown chirp via the Ajith et al. 2007
# PhenomA model (arXiv:gr-qc/0710.2335), with coloured Gaussian noise
# drawn from an analytic aLIGO-design PSD (Damour-Iyer-Sathyaprakash
# form). Two free parameters are exposed to students: chirp mass and
# luminosity distance. Everything else — mass ratio (fixed to equal),
# coalescence time (centre of segment), reference phase, sky location,
# polarization, inclination, detector orientation — is baked in.
#
# The PhenomA polynomial coefficients and the absolute amplitude
# normalisation are deliberately *not* part of the public API. The
# amplitude calibration constant is hand-tuned so that the median of
# the default prior produces a matched-filter SNR of roughly 20 — i.e.
# a "loud" detection in the LIGO-design regime, comparable to GW150914.

# Physical constants (SI), kept private to this module.
_G_SI = 6.67430e-11           # m^3 kg^-1 s^-2
_C_SI = 2.99792458e8          # m / s
_MSUN_KG = 1.98892e30         # kg
_MPC_M = 3.085677581e22       # m

# Derived: solar-mass time-scale GM_sun/c^3 and 1 Mpc in light-travel seconds.
_T_MSUN = _G_SI * _MSUN_KG / _C_SI ** 3  # ~ 4.925e-6 s
_T_MPC = _MPC_M / _C_SI                  # ~ 1.029e14 s

# PhenomA frequency coefficients (Ajith et al. 2007, Table I). Each row
# gives (a, b, c) such that pi * M_total * f_k = a*eta^2 + b*eta + c.
_PHENOM_COEFFS = {
    "merg": (2.9740e-1, 4.4810e-2, 9.5560e-2),
    "ring": (5.9411e-1, 8.9794e-2, 1.9111e-1),
    "sig":  (5.0801e-1, 7.7515e-2, 2.2369e-2),
    "cut":  (8.4845e-1, 1.2848e-1, 2.7299e-1),
}

# Empirical amplitude calibration. Tuned so that matched-filter SNR at
# Mc = 30 M_sun, dL = 500 Mpc is approximately 20 against the default
# PSD/duration/sampling combination below. If you change the duration,
# sampling, or PSD, retune by calling _calibrate_amp_const() once.
_AMP_CALIB = 0.076


def _phenom_freqs(M_total_sec: float) -> tuple[float, float, float, float]:
    """(f_merg, f_ring, sigma, f_cut) in Hz for equal-mass binaries."""
    eta = 0.25
    out = []
    for key in ("merg", "ring", "sig", "cut"):
        a, b, c = _PHENOM_COEFFS[key]
        out.append((a * eta ** 2 + b * eta + c) / (np.pi * M_total_sec))
    return tuple(out)  # type: ignore[return-value]


@dataclass
class GWChirp:
    """Equal-mass PhenomA IMR chirp + analytic aLIGO coloured noise.

    Two-parameter teaching simulator: chirp mass and luminosity distance.

    Parameters
    ----------
    mc_low, mc_high : prior on detector-frame chirp mass [solar masses]
    dl_low, dl_high : prior on luminosity distance [Mpc]
    duration : observation length [s]
    f_sample : sampling rate [Hz]
    f_low : low-frequency cutoff [Hz] (e.g. 20 Hz for advanced LIGO)

    Conventions
    -----------
    * Time series are real, length ``int(duration * f_sample)``.
    * Coalescence is placed at the centre of the segment (``t_c = duration/2``).
    * Strain is dimensionless; FFTs follow ``numpy.fft.rfft`` convention.

    Public API mirrors :class:`BallThrow`:

        sim = GWChirp()
        theta = sim.sample_prior(n)          # (n, 2) array of (Mc, dL)
        x = sim.simulate(theta)              # (n, n_samples) time-domain strain
        s = sim.summary(x)                   # (n, 2) hand-built summary
        mc_g, dl_g, post = sim.true_posterior(x[0])  # 2-D reference posterior
    """

    mc_low: float = 10.0
    mc_high: float = 60.0
    dl_low: float = 100.0
    dl_high: float = 2000.0
    duration: float = 4.0
    f_sample: float = 2048.0
    f_low: float = 20.0

    # Reference-template chirp masses for the 2-scalar matched-filter
    # summary. One light (inspiral-dominated), one heavy (ringdown-
    # dominated); together they disentangle Mc and dL.
    _ref_mc_lo: float = 15.0
    _ref_mc_hi: float = 45.0
    _ref_dl: float = 500.0

    def __post_init__(self) -> None:
        # Precompute and cache normalised reference templates for summary().
        Sn = self.psd(self.freqs)
        h_lo = self._waveform_fd(self._ref_mc_lo, self._ref_dl)
        h_hi = self._waveform_fd(self._ref_mc_hi, self._ref_dl)
        # Inner-product normalisation: <h|h> in rfft-convention, dropping
        # constant prefactors that cancel under the ratio below.
        norm_lo = float(np.sqrt(np.sum(np.abs(h_lo) ** 2 / Sn)))
        norm_hi = float(np.sqrt(np.sum(np.abs(h_hi) ** 2 / Sn)))
        self._h_lo_n = h_lo / norm_lo
        self._h_hi_n = h_hi / norm_hi
        self._Sn_cache = Sn

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(round(self.duration * self.f_sample))

    @property
    def df(self) -> float:
        return 1.0 / self.duration

    @property
    def freqs(self) -> np.ndarray:
        """Positive-frequency rfft bins, shape (n_samples//2 + 1,)."""
        return np.fft.rfftfreq(self.n_samples, d=1.0 / self.f_sample)

    # ------------------------------------------------------------------
    # PSD (analytic aLIGO-design fit)
    # ------------------------------------------------------------------
    def psd(self, f: np.ndarray) -> np.ndarray:
        """One-sided noise PSD S_n(f) in Hz^-1.

        Damour-Iyer-Sathyaprakash analytic fit. Floored above f_low.
        """
        f0 = 215.0
        S0 = 1.0e-49
        x = np.maximum(np.asarray(f, dtype=float), self.f_low) / f0
        return S0 * (x ** -4 + 2.0 + 2.0 * x ** 2)

    # ------------------------------------------------------------------
    # PhenomA waveform (frequency domain, rfft convention)
    # ------------------------------------------------------------------
    def _waveform_fd(self, mc: float, dl: float) -> np.ndarray:
        """Complex PhenomA waveform on ``self.freqs`` (rfft-convention).

        Returned array has the same units/normalisation as
        ``np.fft.rfft`` of a strain time series — i.e. you can add it
        directly to ``rfft(noise_t)`` and inverse-rfft the sum.
        """
        # Equal-mass: M_total = 2^{6/5} * Mc.
        M_tot_sec = (2.0 ** (6.0 / 5.0)) * mc * _T_MSUN
        Mc_sec = mc * _T_MSUN
        dl_sec = dl * _T_MPC

        fmerg, fring, sig, fcut = _phenom_freqs(M_tot_sec)
        f = self.freqs

        # Amplitude at f = f_merg: C = sqrt(5*eta/24)/pi^{2/3} * M_total^{5/6}
        # / dL * f_merg^{-7/6}, in seconds (strain * sec).
        eta = 0.25
        C = (np.sqrt(5.0 * eta / 24.0) * np.pi ** (-2.0 / 3.0)
             * M_tot_sec ** (5.0 / 6.0) / dl_sec
             * fmerg ** (-7.0 / 6.0))

        w = np.zeros_like(f)
        m_insp = (f >= self.f_low) & (f < fmerg)
        m_merg = (f >= fmerg) & (f < fring)
        m_ring = (f >= fring) & (f < fcut)
        w[m_insp] = (f[m_insp] / fmerg) ** (-7.0 / 6.0)
        w[m_merg] = (f[m_merg] / fmerg) ** (-2.0 / 3.0)

        # Ringdown Lorentzian, normalised for continuity at f_ring.
        # L(f) = (1/(2pi)) * sig / ((f - fring)^2 + (sig/2)^2)
        # L(fring) = (1/(2pi)) * sig / (sig/2)^2 = 2/(pi * sig).
        # Merger amplitude at fring is (fring/fmerg)^{-2/3}, so
        # ringdown normalisation factor = (fring/fmerg)^{-2/3} / L(fring).
        L_at_ring = 2.0 / (np.pi * sig)
        w_r_norm = (fring / fmerg) ** (-2.0 / 3.0) / L_at_ring
        L = ((1.0 / (2.0 * np.pi)) * sig
             / ((f[m_ring] - fring) ** 2 + (sig / 2.0) ** 2))
        w[m_ring] = w_r_norm * L

        amp = C * w * _AMP_CALIB

        # Phase: Newtonian SPA (chirp mass only), centred at duration/2.
        with np.errstate(divide="ignore", invalid="ignore"):
            psi_chirp = (3.0 / 128.0
                         * (np.pi * Mc_sec * np.maximum(f, self.f_low))
                         ** (-5.0 / 3.0))
        t_c = self.duration / 2.0
        psi = psi_chirp + 2.0 * np.pi * f * t_c - np.pi / 4.0

        # Convert to rfft convention: rfft of continuous signal -> H_k = h_c(f_k)/dt.
        h_fd = amp * np.exp(-1j * psi) * self.f_sample
        h_fd[~(m_insp | m_merg | m_ring)] = 0.0
        return h_fd

    # ------------------------------------------------------------------
    # Coloured noise
    # ------------------------------------------------------------------
    def _draw_noise_fd(self, rng: np.random.Generator) -> np.ndarray:
        """One rfft-convention realisation of coloured Gaussian noise."""
        f = self.freqs
        sigma_k = np.sqrt(self.n_samples * self.f_sample * self.psd(f) / 4.0)
        return (rng.standard_normal(f.size) * sigma_k
                + 1j * rng.standard_normal(f.size) * sigma_k)

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------
    def sample_prior(
        self, n: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        mc = rng.uniform(self.mc_low, self.mc_high, n)
        dl = rng.uniform(self.dl_low, self.dl_high, n)
        return np.stack([mc, dl], axis=1)

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------
    def simulate(
        self,
        theta: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate strain time series for a batch of (Mc, dL) parameters.

        theta : array of shape (N, 2) or (2,).
        Returns array of shape (N, n_samples) or (n_samples,).
        """
        rng = np.random.default_rng() if rng is None else rng
        theta = np.asarray(theta, dtype=float)
        single = theta.ndim == 1
        theta = np.atleast_2d(theta)
        out = np.empty((theta.shape[0], self.n_samples), dtype=float)
        for i in range(theta.shape[0]):
            h_fd = self._waveform_fd(float(theta[i, 0]), float(theta[i, 1]))
            n_fd = self._draw_noise_fd(rng)
            out[i] = np.fft.irfft(h_fd + n_fd, n=self.n_samples)
        return out[0] if single else out

    # ------------------------------------------------------------------
    # Hand-built summary (2 scalars)
    # ------------------------------------------------------------------
    def summary(self, x: np.ndarray) -> np.ndarray:
        """Two-scalar matched-filter summary against a 2-template mini-bank.

        Both entries are normalised matched-filter responses,

            rho_k = Re < d | h_k > / sqrt( < h_k | h_k > )

        against two fixed reference templates: a light chirp
        (``_ref_mc_lo``) and a heavy one (``_ref_mc_hi``), both at
        ``_ref_dl``. Under pure noise each rho has zero mean and unit
        variance; under a true signal each rho is centred on the
        overlap of that template with the signal — so the *ratio*
        rho_hi / rho_lo carries information about chirp mass, while
        their *magnitude* carries information about luminosity distance.

        x : shape (N, n_samples) or (n_samples,). Returns (N, 2) or (2,).
        """
        single = x.ndim == 1
        x = np.atleast_2d(x)
        Xf = np.fft.rfft(x, axis=-1)
        Sn = self._Sn_cache
        # The inner-product normalisation already absorbs S_n; here we
        # take the real part of < x | h_n > = sum_k conj(X_k) h_n,k / S_n.
        rho_lo = np.real(np.sum(np.conj(Xf) * self._h_lo_n[None, :] / Sn, axis=-1))
        rho_hi = np.real(np.sum(np.conj(Xf) * self._h_hi_n[None, :] / Sn, axis=-1))
        # Rescale to (approximately) unit-variance under noise. For an
        # rfft-convention spectrum, the variance of Re < n | hat h > is
        # N * f_s / 4 (see _draw_noise_fd). Divide it out so students
        # see numbers of O(SNR), not O(1e5).
        scale = np.sqrt(self.n_samples * self.f_sample / 4.0)
        out = np.stack([rho_lo, rho_hi], axis=1) / scale
        return out[0] if single else out

    def simulate_summary(
        self,
        theta: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Convenience: simulate() + summary()."""
        return self.summary(self.simulate(theta, rng=rng))

    # ------------------------------------------------------------------
    # Reference posterior (brute-force 2-D grid)
    # ------------------------------------------------------------------
    def true_posterior(
        self,
        x_obs: np.ndarray,
        n_mc: int = 60,
        n_dl: int = 60,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grid-evaluated reference posterior on (Mc, dL).

        Uses the full matched-filter Gaussian-noise log-likelihood

            log p(d | theta) = -1/2 < d - h(theta) | d - h(theta) >
            < a | b > = 4 * df * sum_f conj(a) * b / S_n(f)

        on an ``n_mc x n_dl`` grid. ``n_mc = n_dl = 60`` takes a few
        seconds on a laptop CPU.

        Returns
        -------
        mc_grid : (n_mc,)
        dl_grid : (n_dl,)
        post    : (n_mc, n_dl), normalised by 2-D trapezoid.
        """
        mc_grid = np.linspace(self.mc_low, self.mc_high, n_mc)
        dl_grid = np.linspace(self.dl_low, self.dl_high, n_dl)
        Xf = np.fft.rfft(np.asarray(x_obs))
        Sn = self.psd(self.freqs)
        log_post = np.empty((n_mc, n_dl))
        # The 4*df prefactor and the |d|^2 self-term are theta-independent,
        # so cancel under log_post -= log_post.max(). We only need the
        # theta-dependent part: 2 Re <d|h> - <h|h>.
        for i, mc in enumerate(mc_grid):
            for j, dl in enumerate(dl_grid):
                Hf = self._waveform_fd(mc, dl)
                inner_dh = np.real(np.sum(np.conj(Xf) * Hf / Sn))
                inner_hh = np.real(np.sum(np.abs(Hf) ** 2 / Sn))
                log_post[i, j] = inner_dh - 0.5 * inner_hh
        log_post -= log_post.max()
        post = np.exp(log_post)
        # 2-D normalisation
        post /= trapezoid(trapezoid(post, dl_grid, axis=1), mc_grid)
        return mc_grid, dl_grid, post

    # ------------------------------------------------------------------
    # Optional: optimal matched-filter SNR (no noise) — useful for
    # calibrating _AMP_CALIB; not used by students.
    # ------------------------------------------------------------------
    def optimal_snr(self, mc: float, dl: float) -> float:
        """Noise-free matched-filter SNR for a signal at (Mc, dL)."""
        Hf = self._waveform_fd(mc, dl)
        Sn = self.psd(self.freqs)
        # <h|h> = 4 * df * sum |H_k/f_s|^2 / S_n  (continuous units)
        rho2 = 4.0 * self.df * np.sum(np.abs(Hf / self.f_sample) ** 2 / Sn)
        return float(np.sqrt(rho2))


__all__ = ["BallThrow", "GWChirp"]
