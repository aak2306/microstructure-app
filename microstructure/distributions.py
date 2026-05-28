"""Particle size distributions.

The app's nominal *Avg. Particle Diameter* is interpreted as the **central
tendency** of the chosen distribution:

- Uniform / Normal: arithmetic mean
- Log-normal: median (= geometric mean = exp(μ_ln))
- Rosin-Rammler (Weibull): characteristic size d₆₃ (63.2% point of the CDF)

This module:
- exposes a stable list of distribution names for the UI to render,
- builds per-call samplers in pixel space (so they slot straight into
  ``place_particles``),
- provides E[r²]/r_nominal² correction factors so the up-front
  particle-count estimate in micro_app.py is unbiased under any
  distribution (paired with the per-shape area factor from
  ``generators``).

All randomness uses ``np.random`` global state to stay consistent with the
rest of the package — the caller seeds, the sampler consumes.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from scipy.special import gamma as gamma_fn

UNIFORM = "Uniform ±%"
LOG_NORMAL = "Log-normal"
NORMAL = "Normal (Gaussian)"
ROSIN_RAMMLER = "Rosin-Rammler"

DISTRIBUTIONS = [UNIFORM, LOG_NORMAL, NORMAL, ROSIN_RAMMLER]


def make_size_sampler(
    distribution: str,
    nominal_rad_px: float,
    *,
    uniform_pct: float = 5.0,
    lognormal_sigma_g: float = 1.5,
    normal_sigma_pct: float = 10.0,
    rr_shape_n: float = 2.5,
) -> Callable[[], float]:
    """Return a zero-arg callable that draws one radius (in pixels) per call.

    Only the parameters relevant to ``distribution`` are read; the rest are
    ignored. Validated parameter ranges live in the UI layer.
    """
    if distribution == UNIFORM:
        p = uniform_pct / 100.0

        def sampler() -> float:
            return nominal_rad_px * (1.0 + np.random.uniform(-p, p))

        return sampler

    if distribution == LOG_NORMAL:
        sigma_ln = math.log(lognormal_sigma_g)

        def sampler() -> float:
            return nominal_rad_px * math.exp(sigma_ln * np.random.randn())

        return sampler

    if distribution == NORMAL:
        sigma_px = nominal_rad_px * (normal_sigma_pct / 100.0)

        def sampler() -> float:
            return nominal_rad_px + sigma_px * np.random.randn()

        return sampler

    if distribution == ROSIN_RAMMLER:
        inv_n = 1.0 / rr_shape_n

        def sampler() -> float:
            # Inverse-CDF sample: F(d) = 1 - exp(-(d/d₆₃)^n).
            # Clip U away from 0 and 1 to avoid log(0) / div-by-zero.
            u = np.random.uniform(1e-12, 1.0 - 1e-12)
            return nominal_rad_px * (-math.log(1.0 - u)) ** inv_n

        return sampler

    raise ValueError(f"Unknown distribution: {distribution!r}")


def expected_r2_factor(
    distribution: str,
    *,
    uniform_pct: float = 5.0,
    lognormal_sigma_g: float = 1.5,
    normal_sigma_pct: float = 10.0,
    rr_shape_n: float = 2.5,
) -> float:
    """Return E[r²] / r_nominal² for the chosen distribution.

    Used to keep the up-front estimate of how many particles to place
    unbiased — particle area scales as r², so the mean particle area is
    π · r_nominal² · this factor. Without it, log-normal and Rosin-Rammler
    runs systematically over/undershoot the target volume fraction.
    """
    if distribution == UNIFORM:
        p = uniform_pct / 100.0
        return 1.0 + p**2 / 3.0  # E[U²] for U ~ Uniform(1-p, 1+p)

    if distribution == LOG_NORMAL:
        sigma_ln = math.log(lognormal_sigma_g)
        # For X = nominal · exp(σ_ln · Z), E[X²]/nominal² = exp(2 σ_ln²).
        return math.exp(2.0 * sigma_ln * sigma_ln)

    if distribution == NORMAL:
        s = normal_sigma_pct / 100.0
        return 1.0 + s * s  # E[X²]/μ² = 1 + (σ/μ)²

    if distribution == ROSIN_RAMMLER:
        # For Weibull with scale d₆₃, E[X^k] = d₆₃^k · Γ(1 + k/n).
        return float(gamma_fn(1.0 + 2.0 / rr_shape_n))

    raise ValueError(f"Unknown distribution: {distribution!r}")
